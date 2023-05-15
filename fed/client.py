# -*- coding: UTF-8 -*-
import torch
from torch.utils.data import DataLoader

from utils.models import get_model
from utils.train import *
from utils.datasets import *


class Client:
    def __init__(self):
        self.model = None
        self.dataset = None

    def set_model(self, model):
        self.model = model

    def set_dataset(self, dataset):
        self.dataset = dataset

    def get_model(self):
        return self.model

    def get_dataset(self):
        return self.dataset

    def train_one_iteration(self):
        pass


class OrdinaryClient(Client):
    def __init__(self, args, dataset=None, idx=None):
        super().__init__()
        # self.model = get_model(args)
        self.loss = get_loss(args.local_loss)
        self.ep = args.local_ep
        self.device = args.device
        self.local_optim = args.local_optim
        self.local_lr = args.local_lr
        self.local_momentum = args.local_momentum
        # self.local_wd = args.local_wd
        self.dataset = DataLoader(DatasetSplit(dataset, idx), batch_size=args.local_bs, shuffle=True)

    def train_one_iteration(self):
        self.model.train()
        self.model = self.model.to(self.device)
        epoch_loss = []
        optim = get_optim(self.model, self.local_optim, self.local_lr, self.local_momentum)
        for _ in range(self.ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.dataset):
                images, labels = images.to(self.device), labels.to(self.device)
                self.model.zero_grad()
                probs = self.model(images)
                loss = self.loss(probs, labels)
                loss.backward()
                optim.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        self.model = self.model.cpu()
        return self.model.state_dict(), len(self.dataset.dataset), sum(epoch_loss) / len(epoch_loss)


def create_clients(args, dataset):
    if args.distribution == 'iid':
        idxs = iid_split(dataset, args.num_clients)
    elif args.distribution == 'dniid':
        idxs = dniid_split(dataset, args.num_clients, args.dniid_param)
    elif args.distribution == 'pniid':
        idxs = pniid_split(dataset, args.num_clients)
    else:
        exit("Unknown Distribution!")
    clients = []
    for idx in idxs.values():
        client = OrdinaryClient(args, dataset, idx)
        clients.append(client)
    return clients
