# coding=gbk
# -*- coding:uft-8 -*-
# test

import torch
import torch.nn as nn
import torch.utils.data as Data
import pandas as pd
import numpy as np
import pickle
from MLPmodel import MLPmodel

if __name__ == '__main__':
    df = pd.read_csv('data.csv', header=None).iloc[100000: 120000, :]
    df = df.astype(np.float32)
    dfX = pd.concat([df.iloc[:, :4], df.iloc[:, 5:]], axis=1).values
    dfY = df.iloc[:, 4].values
    ss = pickle.load(open('data.pkl', 'rb'))
    dfX = ss.transform(dfX)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('data size:')
    trainX = torch.from_numpy(dfX).to(torch.float32).cuda(device)
    print(trainX.size())
    trainY = torch.from_numpy(dfY).to(torch.float32).cuda(device)
    print(trainY.size())
    trainData = Data.TensorDataset(trainX, trainY)
    trainLoader = Data.DataLoader(dataset=trainData, batch_size=128, shuffle=True)

    lossFunction = nn.MSELoss()  # loss function
    model = MLPmodel().cuda()
    model.load_state_dict(torch.load('data.pth'))
    lossLs = []
    for step, (bX, bY) in enumerate(trainLoader):
        output = model(bX).flatten()
        loss = lossFunction(output, bY)
        lossLs.append(loss.item())
    lossValue = round(sum(lossLs) / len(lossLs), 4)
    print(f'loss: {lossValue}')
