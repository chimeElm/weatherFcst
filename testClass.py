# coding=gbk
# -*- coding:uft-8 -*-
# testClass

import torch
import torch.nn as nn
import torch.utils.data as Data
import pandas as pd
import numpy as np
import pickle
from MLPmodel import MLPmodel


class TestClass:
    def __init__(self):
        df = pd.read_csv('data.csv', header=None).iloc[150000: 200000, :]
        df = df.astype(np.float32)
        dfX = pd.concat([df.iloc[:, :4], df.iloc[:, 5:]], axis=1).values
        dfY = df.iloc[:, 4].values
        ss = pickle.load(open('data.pkl', 'rb'))
        dfX = ss.transform(dfX)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        trainX = torch.from_numpy(dfX).to(torch.float32).cuda(device)
        trainY = torch.from_numpy(dfY).to(torch.float32).cuda(device)
        trainData = Data.TensorDataset(trainX, trainY)
        self.trainLoader = Data.DataLoader(dataset=trainData, batch_size=128, shuffle=True)

    def countLoss(self):
        lossFunction = nn.MSELoss()  # loss function
        model = MLPmodel().cuda()
        model.load_state_dict(torch.load('data.pth'))
        lossLs = []
        for step, (bX, bY) in enumerate(self.trainLoader):
            output = model(bX).flatten()
            loss = lossFunction(output, bY)
            lossLs.append(loss.item())
        lossValue = round(sum(lossLs) / len(lossLs), 4)
        print(f'testset loss: {lossValue}')
        return lossValue
