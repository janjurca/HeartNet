#!/usr/bin/env python3

import time
import argparse
import torch

import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

import os
import sys
import math

import shutil

import setproctitle

from nets.vnet import VNet
from utils.dataset import GomezT1
from functools import reduce
import operator
import SimpleITK as sitk
import matplotlib.pyplot as plt


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        nn.init.kaiming_normal(m.weight)
        m.bias.data.zero_()


def datestr():
    now = time.gmtime()
    return '{}{:02}{:02}_{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)


def save_checkpoint(state, is_best, path, prefix, filename='checkpoint.pth.tar'):
    prefix_save = os.path.join(path, prefix)
    name = prefix_save + '_' + filename
    print("Saving:", name)
    torch.save(state, name)
    if is_best:
        shutil.copyfile(name, prefix_save + '_model_best.pth.tar')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=1)
    parser.add_argument('--dataset', default='./Gomez_T1', type=str, help='Dataset Path')
    parser.add_argument('--output', default='./infered/', type=str, help='infered results folder')
    parser.add_argument('--checkpoint', default='./checkpoint.pth.tar', type=str, help='model file')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    print("build vnet")
    model = VNet(elu=False, nll=True)

    if os.path.isfile(args.checkpoint):
        print("=> loading checkpoint '{}'".format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.checkpoint, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
        exit(-1)

    print('  + Number of params: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    if args.cuda:
        model = model.cuda()

    os.makedirs(args.output, exist_ok=True)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    print("loading training set")
    inferenceSet = GomezT1(root=args.dataset, portion=1, resolution=[128, 128, 128])

    model.eval()
    for i in range(len(inferenceSet)):
        (data, target), image = inferenceSet.get(i)
        print(data.size())
        shape = data.size()
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(torch.tensor([data.tolist()]), volatile=True), Variable(target)
        output = model(data)
        _, output = output.data.max(1)  # get the index of the max log-probability

        output = output.view(shape)
        output = output.cpu()
        print(output.size(), data.size())
        output = output.detach().numpy().squeeze()
        for x, _ in enumerate(output):
            for y, _ in enumerate(output[x]):
                for z, _ in enumerate(output[x][y]):
                    output[x][y][z] = output[x][y][z] if output[x][y][z] == 0 else 1
        output = np.array(output, dtype=float)
        img = sitk.GetImageFromArray(output)
        img.SetOrigin(image.GetOrigin())
        sitk.WriteImage(img, args.output+'/image.mhd')
        sitk.Show(img, title="grid using Show function")
        sitk.Show(image, title="grid using Show function")


if __name__ == '__main__':
    main()
