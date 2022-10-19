# coding=gbk
# -*- coding:uft-8 -*-
# build

import torch
import torch.nn as nn
from torch.optim import SGD
import torch.utils.data as Data
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from MLPmodel import MLPmodel
from testClass import TestClass

if __name__ == '__main__':
    df = pd.read_csv('data.csv', header=None).iloc[:150000, :]
    df = df.astype(np.float32)
    dfX = pd.concat([df.iloc[:, :4], df.iloc[:, 5:]], axis=1).values
    dfY = df.iloc[:, 4].values
    ss = StandardScaler()
    dfX = ss.fit_transform(dfX)
    pickle.dump(ss, open('data.pkl', 'wb'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('data size:')
    trainX = torch.from_numpy(dfX).to(torch.float32).cuda(device)
    print(trainX.size())
    trainY = torch.from_numpy(dfY).to(torch.float32).cuda(device)
    print(trainY.size())
    trainData = Data.TensorDataset(trainX, trainY)
    trainLoader = Data.DataLoader(dataset=trainData, batch_size=128, shuffle=True)

    mlp = MLPmodel().cuda()
    optimizer = SGD(mlp.parameters(), lr=0.001)  # define optimizer
    lossFunction = nn.MSELoss()  # loss function

    testObj = TestClass()

    print('start train:')
    trainLossLs = []
    testLossLs = []
    for epoch in range(20):
        lossLs = []
        for step, (bX, bY) in enumerate(trainLoader):
            output = mlp(bX).flatten()
            loss = lossFunction(output, bY)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lossLs.append(loss.item())
        lossValue = round(sum(lossLs) / len(lossLs), 4)
        trainLossLs.append(lossValue)
        print(f'train epoch {epoch + 1}\ntrainset loss: {lossValue}')
        torch.save(mlp.state_dict(), 'data.pth')
        testLoss = testObj.countLoss()
        testLossLs.append(testLoss)

    plt.figure(figsize=(12, 6))
    plt.plot(range(20), trainLossLs, color='r')
    plt.plot(range(20), testLossLs, color='b')
    plt.title('MLPmodel train/test loss per iteration')
    plt.savefig('loss.png')
    plt.show()
