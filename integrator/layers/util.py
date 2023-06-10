import torch
import math


def weight_initializer(weight):
    fan_avg = 0.5 * (weight.shape[-1] + weight.shape[-2])
    std = math.sqrt(1. / fan_avg / 10.)
    a = -2. * std
    b =  2. * std
    torch.nn.init.trunc_normal_(weight, 0., std, a, b)
    return weight

