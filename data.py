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

class AttributeDataset(Dataset):
    def __init__(self, root, split, query_attr_idx=None, transform=None):
        super(AttributeDataset, self).__init__()
        data_path = os.path.join(root, split, "images.npy")
        self.data = np.load(data_path)
        self.data = [T.ToPILImage()(self.data[i]) for i in range(self.data.shape[0])]
        attr_path = os.path.join(root, split, "attrs.npy")
        self.attr = torch.LongTensor(np.load(attr_path))

        attr_names_path = os.path.join(root, "attr_names.pkl")
        with open(attr_names_path, "rb") as f:
            self.attr_names = pickle.load(f)

        self.num_attrs = self.attr.size(1)
        self.set_query_attr_idx(query_attr_idx)
        self.transform = transform

    def set_query_attr_idx(self, query_attr_idx):
        if query_attr_idx is None:
            query_attr_idx = torch.arange(self.num_attrs)

        self.query_attr = self.attr[:, query_attr_idx]

    def __len__(self):
        return self.attr.size(0)

    def __getitem__(self, index):
        image, attr = self.data[index], self.query_attr[index]
        image = self.transform(image)

        return image, attr