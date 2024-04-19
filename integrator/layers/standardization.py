from pylab import *
import polars as pl
import torch
from scipy.spatial import cKDTree
import pandas as pd
import reciprocalspaceship as rs
import torch.nn as nn
from dials.array_family import flex
import numpy as np


class Standardize(nn.Module):
    def __init__(
        self, center=True, feature_dim=7, max_counts=float("inf"), epsilon=1e-6
    ):
        """
        Standardize the data to have a mean = 0 and variance = 1.
        This is based off Welford's algorithm (https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm)

        Args:
            center (bool): Whether to center the data. Defaults to True
            feature_dim (int): Number of feature dimensions. Defaults to 7
            max_counts (float): Maximum number of counts before stopping updates. Defaults to infinity
            epsilon (float): Small value to avoid division by zero. Defaults to 1e-6
        """
        super().__init__()
        self.epsilon = epsilon
        self.center = center
        self.max_counts = max_counts
        self.register_buffer("mean", torch.zeros((1, 1, feature_dim)))
        self.register_buffer("m2", torch.zeros((1, 1, feature_dim)))
        self.register_buffer("count", torch.tensor(0.0))

    @property
    def var(self):
        m2 = torch.clamp(self.m2, min=self.epsilon)
        return m2 / self.count.clamp(min=1)

    @property
    def std(self):
        return torch.sqrt(self.var)

    def update(self, im, mask=None):
        if mask is None:
            k = len(im)
        else:
            k = mask.sum()  # count num of pixels in batch
        self.count += k

        if mask is None:
            diff = im - self.mean
        else:
            diff = (im - self.mean) * mask.unsqueeze(-1)

        self.mean += torch.sum(diff / self.count, dim=(1, 0))

        if mask is None:
            diff *= im - self.mean
        else:
            diff *= (im - self.mean) * mask.unsqueeze(-1)
        self.m2 += torch.sum(diff, dim=(1, 0))

    def standardize(self, im, mask=None):
        if mask is None:
            if self.center:
                return (im - self.mean) / self.std
        else:
            if self.center:
                return ((im - self.mean) * mask.unsqueeze(-1)) / self.std
        return im / self.std

    def forward(self, im, mask, training=True):
        if self.count > self.max_counts:
            training = False
        if training:
            self.update(im, mask)
        return self.standardize(im, mask)
