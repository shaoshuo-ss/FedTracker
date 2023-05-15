# -*- coding: UTF-8 -*-
import copy
import os.path
import time
import numpy as np
import torch
import random
from torch.backends import cudnn
from torch.utils.data import DataLoader
import json

from fed.client import create_clients
from fed.server import FedAvg
from utils.datasets import get_full_dataset
from utils.models import get_model
from utils.test import test_img, test_img_top5
from utils.train import get_optim, gem_train, set_bn_eval
from utils.utils import printf, load_args
from watermark.fingerprint import *
from watermark.watermark import *

if __name__ == '__main__':
    args = load_args()
    # log file path
    log_path = os.path.join(args.save_dir, 'log.txt')
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    # save args
    with open(os.path.join(args.save_dir, "args.txt"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    # load dataset
    train_dataset, test_dataset = get_full_dataset(args.dataset, img_size=(args.image_size, args.image_size))
    # create clients
    clients = create_clients(args, train_dataset)
    # create global model
    global_model = get_model(args)
    if args.pre_train:
        global_model.load_state_dict(torch.load(args.pre_train_path))
    for client in clients:
        client.set_model(copy.deepcopy(global_model))

    # initialize global watermark and local fingerprints
    if args.watermark or args.fingerprint:
        # get weight size
        
        # embed_layer_name = args.embed_layer.split('.')
        # embed_layer = global_model
        # for name in embed_layer_name:
        #     embed_layer = getattr(embed_layer, name)
        # weight_size = embed_layer.weight.shape[0]
        weight_size = get_embed_layers_length(global_model, args.embed_layer_names)

        # generate fingerprints
        local_fingerprints = generate_fingerprints(args.num_clients, args.lfp_length)

        # generate extracting matrix
        extracting_matrices = generate_extracting_matrices(weight_size, args.lfp_length, args.num_clients)

        # generate watermark
        trigger_set = generate_waffle_pattern(args)
        watermark_set = DataLoader(trigger_set, batch_size=args.local_bs, shuffle=True)

    # set seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        random.seed(args.seed)
        cudnn.deterministic = True

    # training
    train_loss = []
    val_loss, val_acc = [], []
    acc_best = None
    client_acc_best = [0 for i in range(args.num_clients)]
    num_clients_each_iter = max(min(args.num_clients, args.num_clients_each_iter), 1)
    for epoch in range(args.start_epochs, args.epochs):
        start_time = time.time()
        local_losses = []
        local_models = []
        local_nums = []

        # adjust learning rate
        for client in clients:
            client.local_lr *= args.lr_decay
        clients_idxs = np.random.choice(range(args.num_clients), num_clients_each_iter, replace=False)

        # train locally
        for idx in clients_idxs:
            current_client = clients[idx]
            local_model, num_samples, local_loss = current_client.train_one_iteration()
            local_models.append(copy.deepcopy(local_model))
            local_losses.append(local_loss)
            local_nums.append(num_samples)
        # aggregation
        global_model.load_global_model(FedAvg(local_models, local_nums), args.device, args.gem)
        # global_model.load_state_dict(FedAvg(local_models, local_nums))

        # print loss
        avg_loss = sum(local_losses) / len(local_losses)
        printf('Round {:3d}, Average loss {:.3f}'.format(epoch, avg_loss), log_path)
        printf('Time: {}'.format(time.time() - start_time), log_path)
        train_loss.append(avg_loss)

        if (epoch + 1) % args.test_interval == 0:
            acc_test, acc_test_top5 = test_img(global_model, test_dataset, args)
            printf("Testing accuracy: Top1: {:.3f}, Top5: {:.3f}".format(acc_test, acc_test_top5), log_path)
            if acc_best is None or acc_test >= acc_best:
                acc_best = acc_test
                if args.save:
                    torch.save(global_model.state_dict(), args.save_dir + "model_best.pth")

        # embed global watermark and local fingerprints
        if args.watermark:
            # embed global watermark
            watermark_optim = get_optim(global_model, args.local_optim, args.lambda1)
            watermark_acc, _ = test_img(global_model, trigger_set, args)
            watermark_loss_func = nn.CrossEntropyLoss()
            watermark_embed_iters = 0
            while watermark_acc <= 98 and watermark_embed_iters <= args.watermark_max_iters:
                global_model.train()
                global_model.to(args.device)
                if args.freeze_bn:
                    global_model.apply(set_bn_eval)
                watermark_embed_iters += 1
                for batch_idx, (images, labels) in enumerate(watermark_set):
                    images, labels = images.to(args.device), labels.to(args.device)
                    global_model.zero_grad()
                    probs = global_model(images)
                    watermark_loss = watermark_loss_func(probs, labels)
                    watermark_loss.backward()
                    if args.gem:
                        global_model = gem_train(global_model)
                    watermark_optim.step()
                watermark_acc, _ = test_img(global_model, trigger_set, args)

            # embed local fingerprints
            if args.fingerprint:
                for client_idx in range(len(clients)):
                    client_fingerprint = local_fingerprints[client_idx]
                    client_model = copy.deepcopy(global_model)
                    embed_layers = get_embed_layers(client_model, args.embed_layer_names)
                    # embed_layer = client_model
                    # for name in embed_layer_name:
                    #     embed_layer = getattr(embed_layer, name)
                    ber, extract_idx = extracting_fingerprints(embed_layers, local_fingerprints, extracting_matrices)
                    count = 0
                    while (extract_idx != client_idx or (client_idx == extract_idx and ber < 0.85))  and count <= args.fingerprint_max_iters:
                        client_grad = calculate_local_grad(embed_layers,
                                                        client_fingerprint,
                                                        extracting_matrices[client_idx])
                        client_grad = torch.mul(client_grad, -args.lambda2)
                        weight_count = 0
                        for embed_layer in embed_layers:
                            weight_length = embed_layer.weight.shape[0]
                            embed_layer.weight = torch.nn.Parameter(torch.add(embed_layer.weight, client_grad[weight_count: weight_count + weight_length]))
                            weight_count += weight_length
                        count += 1
                        ber, extract_idx = extracting_fingerprints(embed_layers, local_fingerprints, extracting_matrices)
                    # ber, extract_idx = extracting_fingerprints(embed_layer, local_fingerprints, extracting_matrices)
                    printf("(Client_idx:{}, Result_idx:{}, BER:{})".format(client_idx, extract_idx, ber), log_path)
                    clients[client_idx].set_model(client_model)
            else:
                for client in clients:
                    client.set_model(copy.deepcopy(global_model))
        else:
            # send global model to clients
            for client in clients:
                client.set_model(copy.deepcopy(global_model))

        # testing
        if (epoch + 1) % args.test_interval == 0:
            if args.watermark:
                # avg_acc = 0.0
                avg_watermark_acc = 0.0
                avg_ber = 0.0
                # max_acc = 0.0
                client_acc = []
                client_acc_top5 = []
                for client_idx in range(args.num_clients):
                    acc, acc_top5 = test_img(clients[client_idx].model, test_dataset, args)
                    watermark_acc, _ = test_img(clients[client_idx].model, trigger_set, args)
                    # avg_acc += acc
                    client_acc.append(acc)
                    client_acc_top5.append(acc_top5)
                    avg_watermark_acc += watermark_acc
                    if args.fingerprint:
                        embed_layers = get_embed_layers(clients[client_idx].model, args.embed_layer_names)
                        ber, extract_idx = extracting_fingerprints(embed_layers, local_fingerprints, extracting_matrices)
                        avg_ber += ber
                    if acc >= client_acc_best[client_idx]:
                        client_acc_best[client_idx] = acc
                        if args.save:
                            torch.save(clients[client_idx].get_model().state_dict(), args.save_dir + "model_" + str(client_idx) + ".pth")
                avg_acc = np.mean(client_acc)
                max_acc = np.max(client_acc)
                min_acc = np.min(client_acc)
                median_acc = np.median(client_acc)
                low_acc = np.percentile(client_acc, 25)
                high_acc = np.percentile(client_acc, 75)
                avg_acc_top5 = np.mean(client_acc_top5)
                max_acc_top5 = np.max(client_acc_top5)
                min_acc_top5 = np.min(client_acc_top5)
                median_acc_top5 = np.median(client_acc_top5)
                low_acc_top5 = np.percentile(client_acc_top5, 25)
                high_acc_top5 = np.percentile(client_acc_top5, 75)
                avg_watermark_acc = avg_watermark_acc / args.num_clients
                avg_ber = avg_ber / args.num_clients
                printf("Clients Average Testing accuracy: {:.2f}".format(avg_acc), log_path)
                printf("Clients Quantile Testing accuracy, Top1: {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}".format(min_acc, low_acc, median_acc, high_acc, max_acc), log_path)
                printf("Clients Quantile Testing accuracy, Top5: {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}".format(min_acc_top5, low_acc_top5, median_acc_top5, high_acc_top5, max_acc_top5), log_path)
                printf("Watermark Average Testing accuracy:{:.2f}".format(avg_watermark_acc), log_path)
                if args.fingerprint:
                    printf("Average BER: {:.4f}".format(avg_ber), log_path)

    printf("Best Acc of Global Model:" + str(acc_best), log_path)
    if args.watermark:
        printf("Clients' Best Acc:", log_path)
        for client_idx in range(args.num_clients):
            printf(client_acc_best[client_idx], log_path)
        avg_acc = np.mean(client_acc_best)
        max_acc = np.max(client_acc_best)
        min_acc = np.min(client_acc_best)
        median_acc = np.median(client_acc_best)
        low_acc = np.percentile(client_acc_best, 25)
        high_acc = np.percentile(client_acc_best, 75)
        # avg_watermark_acc = avg_watermark_acc / args.num_clients
        # avg_ber = avg_ber / args.num_clients
        printf("Clients Average Testing accuracy: {:.2f}".format(avg_acc), log_path)
        printf("Clients Quantile Testing accuracy: {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}".format(min_acc, low_acc, median_acc, high_acc, max_acc), log_path)
        # printf("Watermark Average Testing accuracy:{:.2f}".format(avg_watermark_acc), log_path)
        # printf("Average BER: {:.4f}".format(avg_ber), log_path)
    if args.save:
        torch.save(global_model.state_dict(),
                   args.save_dir + "model_last_epochs_" + str((args.epochs + args.start_epochs)) + ".pth")
