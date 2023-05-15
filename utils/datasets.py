# -*- coding: UTF-8 -*-

import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms, datasets
from torchvision.datasets import CIFAR10, MNIST, ImageFolder, CIFAR100
import numpy as np
import os
import sys
from PIL import Image


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.Train = train
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")

        if (self.Train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.Train)

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()

        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images;

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(train_dir, d))]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        # self.idx_to_class = {i:self.val_img_to_class[images[i]] for i in range(len(images))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        self.images = []
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        if Train:
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            item = (path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
                        self.images.append(item)

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        img_path, tgt = self.images[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, tgt


def get_full_dataset(dataset_name, img_size=(32, 32)):
    if dataset_name == 'mnist':
        train_dataset = MNIST('./data/mnist/', train=True, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Resize(img_size),
                                  # transforms.RandomHorizontalFlip(),
                                  transforms.Normalize((0.1307,), (0.3081,))
                              ]))
        test_dataset = MNIST('./data/mnist/', train=False, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Resize(img_size),
                                 transforms.Normalize((0.1307,), (0.3081,))
                             ]))
    elif dataset_name == 'cifar10':
        train_dataset = CIFAR10('./data/cifar10/', train=True, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Pad(4, padding_mode="reflect"),
                                    transforms.RandomCrop(img_size),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.244, 0.262))
                                ]))
        test_dataset = CIFAR10('./data/cifar10/', train=False, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Resize(img_size),
                                   transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.244, 0.262))
                               ]))
    elif dataset_name == 'cifar100':
        train_dataset = CIFAR100('./data/cifar100/', train=True, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Pad(4, padding_mode="reflect"),
                                    transforms.RandomCrop(img_size),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.244, 0.262))
                                ]))
        test_dataset = CIFAR100('./data/cifar100/', train=False, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Resize(img_size),
                                   transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.244, 0.262))
                               ]))
    elif dataset_name == 'imagenet10':
        full_dataset = ImageFolder('./data/imagenet10/train_set/',
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Resize(img_size),
                                        transforms.Normalize((0.52283615, 0.47988218, 0.40605107), (0.29770654, 0.2888402, 0.31178293))
                                    ]))
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, test_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    elif dataset_name == 'tinyimagenet':
        train_dataset = TinyImageNet('./data/tiny-imagenet-200/', train=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Pad(4, padding_mode="reflect"),
                                    transforms.RandomCrop(img_size),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.Resize(img_size),
                                    transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))
                                ]))
        test_dataset = TinyImageNet('./data/tiny-imagenet-200/', train=False,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Resize(img_size),
                                    transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))
                                ]))
    else:
        exit("Unknown Dataset")
    return train_dataset, test_dataset


def iid_split(dataset, num_clients):
    """
    Split I.I.D client data
    :param dataset:
    :param num_clients:
    :return: dict of image indexes
    """

    dataset_len = len(dataset)
    num_items = dataset_len // num_clients
    dict_clients = dict()
    all_idxs = [i for i in range(dataset_len)]
    for i in range(num_clients):
        dict_clients[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_clients[i])
    return dict_clients


def dniid_split(dataset, num_clients, param=0.8):
    """
    Using Dirichlet distribution to sample non I.I.D client data
    :param dataset:
    :param num_clients:
    :param param: parameter used in Dirichlet distribution
    :return: dict of image indexes
    """
    dataset_len = len(dataset)
    dataset_y = np.array(dataset.targets)
    labels = set(dataset_y)
    sorted_idxs = dict()
    for label in labels:
        sorted_idxs[label] = []

    # sort indexes by labels
    for i in range(dataset_len):
        sorted_idxs[dataset_y[i]].append(i)

    for label in labels:
        sorted_idxs[label] = np.array(sorted_idxs[label])

    # initialize the clients' dataset dict
    dict_clients = dict()
    for i in range(num_clients):
        dict_clients[i] = None
    # split the dataset separately
    for label in labels:
        idxs = sorted_idxs[label]
        sample_split = np.random.dirichlet(np.array(num_clients * [param]))
        accum = 0.0
        num_of_current_class = idxs.shape[0]
        for i in range(num_clients):
            client_idxs = idxs[int(accum * num_of_current_class):
                               min(dataset_len, int((accum + sample_split[i]) * num_of_current_class))]
            if dict_clients[i] is None:
                dict_clients[i] = client_idxs
            else:
                dict_clients[i] = np.concatenate((dict_clients[i], client_idxs))
            accum += sample_split[i]
    return dict_clients


def pniid_split(dataset, num_clients, num_of_shards_each_clients=2):
    """
    Simulate pathological non I.I.D distribution
    :param dataset:
    :param num_clients:
    :param num_of_shards_each_clients:
    :return:
    """
    dataset_len = len(dataset)
    dataset_y = np.array(dataset.targets)

    sorted_idxs = np.argsort(dataset_y)

    size_of_each_shards = dataset_len // (num_clients * num_of_shards_each_clients)
    per = np.random.permutation(num_clients * num_of_shards_each_clients)
    dict_clients = dict()
    for i in range(num_clients):
        idxs = np.array([])
        for j in range(num_of_shards_each_clients):
            idxs = np.concatenate((idxs, sorted_idxs[per[num_of_shards_each_clients * i + j] * size_of_each_shards:
                                   min(dataset_len, (per[num_of_shards_each_clients * i + j] + 1) * size_of_each_shards)]))
        dict_clients[i] = idxs
    return dict_clients


if __name__ == '__main__':
    train_set, test_set = get_full_dataset('imagenet10', (224, 224))
    # print(train_set.class_to_idx)
    print(len(train_set))
    print(len(test_set))
    # print(train_set.targets)
