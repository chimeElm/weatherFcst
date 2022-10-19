# coding=gbk
# -*- coding:uft-8 -*-
# predict

import numpy
from xarray import open_dataset
import os
import torch
import torch.nn as nn
import numpy as np
import pickle


def inp(d, x, y):
    data = []
    names = [
        'Relative humidity',
        'Specific humidity',
        'Vertical velocity',
        'Temperature',
        'V component of wind',
        'U component of wind',
        'Potential vorticity',
        'Divergence',
        'Geopotential Height',
        'Convective available potential energy',
        'Total column water',
        'Total cloud cover',
        'Low cloud cover',
        'Large-scale precipitation',
        'Total precipitation',
        'Maximum temperature at 2 metres in the last 3 hours',
        'Minimum temperature at 2 metres in the last 3 hours',
        'Sea surface temperature',
        '2 metre dewpoint temperature',
        '2 metre temperature',
        '10 metre U wind component',
        '10 metre V wind component'
    ]
    for name in names:
        array = d.data_vars[name].values
        if len(array.shape) == 2:
            data.append(array[x][y])
        else:
            for z in range(5):
                data.append(array[x][y][z])
    return data


class MLPmodel(nn.Module):
    def __init__(self):
        super(MLPmodel, self).__init__()
        # first hidden layer
        self.h1 = nn.Linear(in_features=62, out_features=120, bias=True)
        self.a1 = nn.ReLU()
        # second hidden layer
        self.h2 = nn.Linear(in_features=120, out_features=60, bias=False)
        self.a2 = nn.ReLU()
        # third hidden layer
        self.h3 = nn.Linear(in_features=60, out_features=20, bias=False)
        self.a3 = nn.ReLU()
        # regression predict layer
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


if __name__ == '__main__':
    print('start predict:')
    ss = pickle.load(open('data.pkl', 'rb'))
    mlp = MLPmodel().cuda()
    mlp.load_state_dict(torch.load('data.pth'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fileLs = os.listdir('./dataset/test')
    for order in fileLs:
        path = './Pred_temperature/' + order
        if not os.path.exists(path):
            os.mkdir(path)
        inLs = os.listdir('./dataset/test/' + order)
        star = eval(inLs[0][:2])
        for hour in range(9):
            hour += 1
            predY = []
            fileIn = f'./dataset/test/{order}/grid_inputs_0{hour}.nc'
            dsIn = open_dataset(fileIn)
            for row in open(f'./dataset/test/{order}/ji_loc_inputs_0{hour}.txt', 'r', encoding='utf-8').readlines():
                xl, yl = eval(row.strip().split()[0]), eval(row.strip().split()[1])
                dataEx = [[star, hour, xl + 1, yl + 1] + inp(dsIn, xl, yl)]
                predX = numpy.array(dataEx).astype(np.float32)
                predX = ss.transform(predX)
                predX = torch.from_numpy(predX[0]).to(torch.float32).cuda(device)
                output = mlp(predX).flatten()
                predY.append(str(output.item()))
            pred = '\n'.join(predY)
            open(f'{path}/pred_0{hour}.txt', 'w', encoding='utf-8').write(pred)
            print(order, hour)
