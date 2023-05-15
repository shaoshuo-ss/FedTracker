# -*- coding: UTF-8 -*-
import torch
import quadprog
import numpy as np


def get_optim(model, optim, lr=0.1, momentum=0):
    if optim == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    else:
        exit("Unknown Optimizer!")


def get_loss(loss):
    if loss == "CE":
        return torch.nn.CrossEntropyLoss()
    elif loss == "MSE":
        return torch.nn.MSELoss()
    else:
        exit("Unknown Loss")


def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.Tensor(x).view(-1, 1))
    return gradient


def store_grad(pp, grads, grad_dims, tid):
    # store the gradients
    grads[:, tid].fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en, tid].copy_(param.grad.data.view(-1))
        cnt += 1


def overwrite_grad(pp, newgrad, grad_dims):
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1


def gem_train(model):
    for name, param in model.named_parameters():
        grads = param.grad
        grad_size = grads.data.size()
        if grads is not None:
            memory = model.memory[name]
            # check
            dotp = torch.mul(grads.view(-1), memory.view(-1))
            if dotp.sum() < 0:
                newgrads = project2cone2(grads.view(-1).unsqueeze(1), memory.view(-1, 1))
                param.grad.data.copy_(newgrads.view(grad_size))
    return model


def set_bn_eval(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval()
