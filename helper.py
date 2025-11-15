import logging
import sys
import os
import numpy as np
import copy

import torch
import torchvision
import matplotlib.pyplot as plt
import wandb
import torch.distributed as dist
from reptrix import alpha, rankme, lidar
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

def ddp_active() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1



# --------- small DDP helpers ----------
def ddp_active():
    return dist.is_available() and dist.is_initialized()

def is_rank0():
    return (not ddp_active()) or dist.get_rank() == 0

def ddp_barrier():
    if ddp_active():
        dist.barrier()

# --------- feature extraction & cache ----------
@torch.no_grad()
def _extract_and_cache(encoder, loader, device, cache_dir, split):
    os.makedirs(cache_dir, exist_ok=True)
    X_chunks, A_chunks = [], []

    encoder.eval()
    for data, attr in loader:
        data = data.to(device, non_blocking=True)
        x = encoder.backbone(data)                           # [B,T,C] or [B,C]
        # x = feats.mean(dim=1) if feats.dim() == 3 else feats
        x = F.normalize(x, dim=1)
        X_chunks.append(x.cpu())
        A_chunks.append(attr.cpu())

    X = torch.cat(X_chunks, dim=0)                   # [N, D]
    A = torch.cat(A_chunks, dim=0)                   # [N, K]
    torch.save(X, os.path.join(cache_dir, f"{split}_X.pt"))
    torch.save(A, os.path.join(cache_dir, f"{split}_A.pt"))

def _load_cached(cache_dir, split, map_location="cpu"):
    X = torch.load(os.path.join(cache_dir, f"{split}_X.pt"), map_location=map_location)
    A = torch.load(os.path.join(cache_dir, f"{split}_A.pt"), map_location=map_location)
    return X, A

# --------- main (precompute + train probes only) ----------
def compute_attribute_wise_acc_cached(
    device,
    encoder,
    lm_probe_digit,
    lm_probe_color,
    unbiased_train_loader,
    unbiased_test_loader,
    unbiased_train_sampler=None,   # not needed anymore, but kept for signature compatibility
    unbiased_test_sampler=None,    # not needed
    cache_dir=None,
    epochs=100,
    batch_size=4096,
    lr=5e-3,
    weight_decay=1e-4,
    delete_cache=True,
):
    """
    Precompute encoder features once to cache_dir, then train digit/color probes on cached tensors.
    Assumes attr[:,0] is digit labels and attr[:,1] is color labels.
    """

    # 1) Pick a cache dir
    if cache_dir is None:
        job_id   = os.getenv("SLURM_JOB_ID")
        cache_dir = "/leonardo_work/EUHPC_D27_070/cache/" + str(job_id)
        # cache_dir = "/vol/research/project_storage/cache/"
    os.makedirs(cache_dir, exist_ok=True)

    # 2) Extract & cache only on rank0; others wait then load
    prev_training = encoder.training
    if is_rank0():
        # Cache if missing (idempotent)
        need_train = not (os.path.exists(os.path.join(cache_dir, "train_X.pt")) and
                          os.path.exists(os.path.join(cache_dir, "train_A.pt")))
        need_test  = not (os.path.exists(os.path.join(cache_dir, "test_X.pt"))  and
                          os.path.exists(os.path.join(cache_dir, "test_A.pt")))
        if need_train:
            _extract_and_cache(encoder, unbiased_train_loader, device, cache_dir, "train")
        if need_test:
            _extract_and_cache(encoder, unbiased_test_loader,  device, cache_dir, "test")
    ddp_barrier()

    # 3) Load cached tensors (CPU)
    Xtr, Atr = _load_cached(cache_dir, "train", map_location="cpu")
    Xte, Ate = _load_cached(cache_dir, "test",  map_location="cpu")

    # 4) Build dataloaders from cached tensors
    train_both  = TensorDataset(Xtr, Atr)  # labels shape [N,2]
    train_loader_cached = DataLoader(
        train_both, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=True
    )

    # 5) Optimizer over probe params only
    lm_probe_digit.to(device)
    lm_probe_color.to(device)
    lm_probe_digit.train()
    lm_probe_color.train()

    optimizer = torch.optim.Adam(
        list(lm_probe_digit.parameters()) + list(lm_probe_color.parameters()),
        lr=lr, weight_decay=weight_decay
    )

    # 6) Train probes (no encoder calls now)
    for epoch in range(epochs):
        batch_losses_digit = []
        batch_losses_color = []

        for xb, ab in train_loader_cached:
            xb = xb.to(device, non_blocking=True)
            y_digit = ab[:, 0].to(device, non_blocking=True)
            y_color = ab[:, 1].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            # Your probes return (logits, loss) when given (x, label)
            _, loss_digit = lm_probe_digit(xb, y_digit)
            _, loss_color = lm_probe_color(xb, y_color)
            loss = loss_digit + loss_color
            loss.backward()
            optimizer.step()
            batch_losses_digit.append(loss_digit.item())
            batch_losses_color.append(loss_color.item())

        if os.getenv("RANK", "0") == "0":
            wandb.log({"eval_lm_probe/train_loss_digit": sum(batch_losses_digit) / max(1, len(batch_losses_digit))})
            wandb.log({"eval_lm_probe/train_loss_color": sum(batch_losses_color) / max(1, len(batch_losses_color))})

    # 7) Eval (again, no encoder calls)
    lm_probe_digit.eval()
    lm_probe_color.eval()

    test_loader_cached = DataLoader(
        TensorDataset(Xte, Ate),
        batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True
    )

    correct_digit = torch.tensor(0, device=device, dtype=torch.long)
    correct_color = torch.tensor(0, device=device, dtype=torch.long)
    total         = torch.tensor(0, device=device, dtype=torch.long)

    with torch.inference_mode():
        for xb, ab in test_loader_cached:
            xb = xb.to(device, non_blocking=True)
            y_digit = ab[:, 0].to(device, non_blocking=True).long()
            y_color = ab[:, 1].to(device, non_blocking=True).long()

            logit_digit, _ = lm_probe_digit(xb, y_digit)   # keep signature consistent
            logit_color, _ = lm_probe_color(xb, y_color)

            pred_digit = logit_digit.argmax(dim=1)
            pred_color = logit_color.argmax(dim=1)

            correct_digit += (pred_digit == y_digit).sum()
            correct_color += (pred_color == y_color).sum()
            total         += y_digit.size(0)

    if ddp_active():
        dist.all_reduce(correct_digit, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_color, op=dist.ReduceOp.SUM)
        dist.all_reduce(total,         op=dist.ReduceOp.SUM)

    acc_digit = (correct_digit.float() / total.float()).item()
    acc_color = (correct_color.float() / total.float()).item()

    # Restore encoder mode (even though we didn't use it during training/eval)
    encoder.train(prev_training)

    ddp_barrier()  # ensure all ranks finished reading cached tensors
    if delete_cache and is_rank0():
        for name in ("train_X.pt", "train_A.pt", "test_X.pt", "test_A.pt"):
            p = os.path.join(cache_dir, name)
            try:
                os.remove(p)
            except FileNotFoundError:
                pass
            except Exception as e:
                print(f"[cache cleanup] could not remove {p}: {e}")
        # try removing the dir if it’s now empty
        try:
            os.rmdir(cache_dir)
        except OSError:
            # non-empty or in use; ignore
            pass
    ddp_barrier()  # make sure deletion is complete before returning (optional)

    return acc_digit, acc_color, lm_probe_digit, lm_probe_color


def compute_other_metrics(encoder, unbiased_test_loader, unbiased_test_loader_full, device, eps=0.0,power=1, rtol=1e-12, num_augmentations=25):
    encoder.eval()
    features_list = []
    transform_ssl = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        torchvision.transforms.RandomGrayscale(p=0.2),
        torchvision.transforms.RandomResizedCrop(
            224, scale=(0.8, 1.0),
            ratio=(0.75, (4 / 3)),
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    for data, attr in unbiased_test_loader_full:
        data = data.to(device)
        with torch.no_grad():
            features = encoder(data)
        features_list.append(features.mean(dim=1))
    latents = torch.cat(features_list, dim=0)

    metric_alpha = alpha.get_alpha(latents)
    metric_rankme = rankme.get_rankme(latents)
    features_list = []
    for data, attr in unbiased_test_loader:
        data = torch.cat([transform_ssl(data) for _ in range(num_augmentations)], dim=0)
        data = data.to(device)
        with torch.no_grad():
            features = encoder(data)
        features_list.append(features.mean(dim=1).unsqueeze(1)) ##pooling on token dimension
    all_representations_ssl = torch.cat(features_list, dim=0)
    all_representations_ssl =  all_representations_ssl.reshape(num_augmentations, -1, features.shape[1]).transpose(1, 0)
    num_samples = all_representations_ssl.shape[0] // num_augmentations
    ramdom_sample_index = torch.randperm(all_representations_ssl.shape[0])[:num_samples]
    subsampled_representations_ssl = all_representations_ssl[ramdom_sample_index]
    metric_lidar = lidar.get_lidar(subsampled_representations_ssl.view(subsampled_representations_ssl.shape[0], num_augmentations, -1), num_samples,
                                   num_augmentations,
                                   del_sigma_augs=0.001)
    return metric_alpha, metric_rankme, metric_lidar