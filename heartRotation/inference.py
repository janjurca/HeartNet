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

from nets.vnet import VNet, VNetRegression
from utils.dataset import GomezT1Rotation
from functools import reduce
import operator
import SimpleITK as sitk
import matplotlib.pyplot as plt


def inference(dataset, checkpoint):
    cuda = torch.cuda.is_available()
    model = VNetRegression(elu=False, nll=True)

    print("=> loading checkpoint '{}'".format(checkpoint))
    checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])

    if cuda:
        model = model.cuda()

    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    print("loading training set")
    inferenceSet = GomezT1Rotation(root=dataset, portion=1, resolution=[128, 128, 128])

    model.eval()
    for i in range(len(inferenceSet)):
        (data, target), image, gt = inferenceSet.get(i)
        shape = data.size()

        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(torch.tensor([data.tolist()]), volatile=True), Variable(target)
        output = model(data)

        output = output.view(shape)
        output = output.cpu()
        output = output.detach().numpy().squeeze()
        output = np.array(output, dtype=float)

        gt.setData(output)
        yield image, gt
