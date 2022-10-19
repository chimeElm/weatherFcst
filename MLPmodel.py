# coding=gbk
# -*- coding:uft-8 -*-
# MLPmodel

import torch.nn as nn


class MLPmodel(nn.Module):
    def __init__(self):
        super(MLPmodel, self).__init__()
        self.h1 = nn.Linear(in_features=62, out_features=100, bias=True)
        self.a1 = nn.ReLU()
        self.h2 = nn.Linear(in_features=100, out_features=50, bias=False)
        self.a2 = nn.Sigmoid()
        self.h3 = nn.Linear(in_features=50, out_features=20, bias=False)
        self.a3 = nn.ReLU()
        self.regression = nn.Linear(in_features=20, out_features=1)

    def forward(self, x):
        x = self.h1(x)
        x = self.a1(x)
        x = self.h2(x)
        x = self.a2(x)
        x = self.h3(x)
        x = self.a3(x)
        outp = self.regression(x)
        return outp
