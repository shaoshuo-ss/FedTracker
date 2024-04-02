# -*- coding: UTF-8 -*-
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F


def test_img(net_g, datatest, args):
    net_g.eval()
    net_g.to(args.device)
    # testing
    correct = 0
    correct_top5 = 0
    data_loader = DataLoader(datatest, batch_size=args.test_bs)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g(data)
        # get the index of the max log-probability
        _, y_pred = torch.max(log_probs.data, 1)
        correct += (y_pred == target).sum().item()
        _, pred = log_probs.topk(5, 1, True, True)
        target_resize = target.view(-1, 1)
        correct_top5 += torch.eq(pred, target_resize).sum().float().item()

    accuracy = 100.00 * correct / len(data_loader.dataset)
    accuracy_top5 = 100.00 * correct_top5 / len(data_loader.dataset)
    net_g.cpu()
    return accuracy, accuracy_top5

