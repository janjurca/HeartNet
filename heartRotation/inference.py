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


def inference(dataset, checkpoint, planes=["sa", "ch4", "ch2"]):
    cuda = torch.cuda.is_available()
    model = VNetRegression(elu=False, nll=True, outCH=len(planes))

    print("=> loading checkpoint '{}'".format(checkpoint))
    checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])

    if cuda:
        model = model.cuda()

    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    model.eval()
    for i in range(len(dataset)):
        (data, _), image, gtsa, gtch4, gtch2 = dataset.get(i)
        gtsa = gtsa.clone()
        gtch4 = gtch4.clone()
        gtch2 = gtch2.clone()
        if cuda:
            data = data.cuda()

        with torch.no_grad():
            data = torch.tensor([data.tolist()])
            output = model(data)

        output = output.view([len(planes), 128, 128, 128])
        output = output.cpu()
        output = output.detach().numpy()
        if len(planes) == 1:
            if "sa" in planes:
                sa = np.array(output[0], dtype=float)
                gtsa.setData(sa)
            if "ch4" in planes:
                ch4 = np.array(output[0], dtype=float)
                gtch4.setData(ch4)
            if "ch2" in planes:
                ch2 = np.array(output[0], dtype=float)
                gtch2.setData(ch2)
        elif len(planes) == 3:
            sa = np.array(output[0], dtype=float)
            ch4 = np.array(output[1], dtype=float)
            ch2 = np.array(output[2], dtype=float)
            gtsa.setData(sa)
            gtch4.setData(ch4)
            gtch2.setData(ch2)
        else:
            raise Exception("Planes are badly defined.")

        yield image, gtsa, gtch4, gtch2
