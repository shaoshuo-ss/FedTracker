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
        maxk = max((1,5))
        _, pred = log_probs.topk(maxk, 1, True, True)
        target_resize = target.view(-1, 1)
        correct_top5 += torch.eq(pred, target_resize).sum().float().item()

    # test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    accuracy_top5 = 100.00 * correct_top5 / len(data_loader.dataset)
    net_g.cpu()
    return accuracy, accuracy_top5


def test_img_top5(model, dataset, args):
    model.eval()
    model.to(args.device)
    correct = 0

    data_loader = DataLoader(dataset, batch_size=args.test_bs)
    
    for x, y in data_loader:
        x, y = x.to(args.device), y.to(args.device)
        with torch.no_grad():
            logits = model(x)
            maxk = max((1,5))
            y_resize = y.view(-1,1)
            _, pred = logits.topk(maxk, 1, True, True)
            correct += torch.eq(pred, y_resize).sum().float().item()

    model.cpu()
    return 100.00 * correct / len(data_loader.dataset)
