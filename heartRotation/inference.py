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
from utils.dataset import GomezT1Rotation
from functools import reduce
import operator
import SimpleITK as sitk
import matplotlib.pyplot as plt

fig, (original, gt, mask) = None, (None, None, None)


class VolumeImage:
    def __init__(self, image, ax, title="Nic") -> None:
        global fig, original, mask
        self.image = image
        self.data = sitk.GetArrayFromImage(self.image)
        self.ax = ax
        self.index = int(len(self.data)/2)
        self.ax_data = self.ax.imshow(self.data[self.index])

        def onScroll(event):
            if event.button == "up":
                self.index += 1
            if event.button == "down":
                self.index -= 1
            self.index = 0 if self.index < 0 else (len(self.data) - 1 if self.index > len(self.data) else self.index)
            self.ax.set_title(f"Slice: {self.index}")
            self.ax_data.set_data(self.data[self.index])
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect('scroll_event', onScroll)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        nn.init.kaiming_normal(m.weight)
        m.bias.data.zero_()


def datestr():
    now = time.gmtime()
    return '{}{:02}{:02}_{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)


def main():
    global fig, original, mask
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
    inferenceSet = GomezT1Rotation(root=args.dataset, portion=1, resolution=[128, 128, 128])

    model.eval()
    for i in range(len(inferenceSet)):
        (data, target), image, gt = inferenceSet.get(i)
        shape = data.size()

        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(torch.tensor([data.tolist()]), volatile=True), Variable(target)
        output = model(data)
        _, output = output.data.max(1)  # get the index of the max log-probability

        output = output.view(shape)
        output = output.cpu()
        output = output.detach().numpy().squeeze()
        output = np.array(output, dtype=float)

        img = sitk.GetImageFromArray(output)
        img.SetOrigin(image.image.GetOrigin())
        #sitk.WriteImage(img, args.output+'/image.mhd')
        sitk.Show(img, title="inffered")
        sitk.Show(image.image, title="orig")
        sitk.Show(gt.image, title="gt")

        #fig, (original, gt, mask) = plt.subplots(1, 3)
        #VolumeImage(image.image, original)
        #VolumeImage(gt.image, gt)
        #VolumeImage(img, mask)
        # plt.show()
        exit(0)


if __name__ == '__main__':
    main()
