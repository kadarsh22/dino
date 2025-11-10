import os
import subprocess
import time

import numpy as np
import random

from logging import getLogger
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torchvision import datasets

import torch
import torchvision.transforms as T
import glob
from torch.utils.data.dataset import Dataset
from torch.utils.data import random_split
from glob import glob
from PIL import Image
import pickle


_GLOBAL_SEED = 0
logger = getLogger()

def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def make_cifar10(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=4,
    split='test',
    root_path=None,
    image_folder=None,
    training=True,
    copy_data=False,
    drop_last=False,
    subset_file=None
):

    dataset = datasets.CIFAR10(
        root=root_path,
        train=True,
        transform=transform,
        download=False,
    )


    unbiased_dataset = datasets.CIFAR10(
        root=root_path,
        train=False,
        transform=transform,
        download=False,
    )

    train_size = int(0.8 * len(unbiased_dataset))
    test_size = int(0.2 * len(unbiased_dataset))

    g = torch.Generator()
    g.manual_seed(0)  # or your global seed
    unbiased_train_dataset, unbiased_test_dataset = random_split(unbiased_dataset,
                                                                 [train_size, test_size], generator=g)


    return dataset, unbiased_train_dataset, unbiased_test_dataset