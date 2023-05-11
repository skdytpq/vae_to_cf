import numpy as np
import os
import torch
import torch.nn as nn


class EASE(nn.Module):

    def __init__(self, input_dim):

    self.weight = nn.Parameter(torch.rand(input_dim, input_dim).fill_diagonal_(0))
    self.sigmoid = nn.Sigmoid()

    def symmetric(X):

      return X.triu() + X.triu(1).transpose(-1, -2)

    def forward(self, x):

      A = symmetric(self.weight)

      return self.sigmoid(x @ A)