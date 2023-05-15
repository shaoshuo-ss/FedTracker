# -*- coding: UTF-8 -*-
import copy

import torch


def FedAvg(models, nums):
    model_avg = copy.deepcopy(models[0])
    total = sum(nums)
    for k in model_avg.keys():
        model_avg[k] = torch.div(model_avg[k], total / nums[0])
        for i in range(1, len(models)):
            model_avg[k] += torch.div(models[i][k], total / nums[i])
    return model_avg
