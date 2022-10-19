# coding=gbk
# -*- coding:uft-8 -*-
# extract

from xarray import open_dataset
import os


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


if __name__ == '__main__':
    csv = open('data.csv', 'w', encoding='utf-8-sig')
    fileLs = os.listdir('./dataset/train')[:100]
    for order in fileLs:
        inLs = os.listdir('./dataset/train/' + order)
        star = eval(inLs[0][:2])
        for hour in range(9):
            hour += 1
            fileIn = f'./dataset/train/{order}/grid_inputs_0{hour}.nc'
            dsIn = open_dataset(fileIn)
            fileOt = f'./dataset/train/{order}/obs_grid_temp0{hour}.nc'
            dsOt = open_dataset(fileOt)
            temps = dsOt.data_vars['obs_temperature'].values
            for xl in range(69):
                for yl in range(73):
                    temp = temps[xl][yl]
                    if temp != -99999:
                        dataEx = [star, hour, xl + 1, yl + 1, temp] + inp(dsIn, xl, yl)
                        dataEx = [str(da) for da in dataEx]
                        csv.write(','.join(dataEx) + '\n')
            print(order, hour)
    csv.close()
