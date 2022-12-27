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


from nets.vnet import VNet, VNetRegression
from utils.dataset import GomezT1Rotation
from functools import reduce
import operator


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
    parser.add_argument('--nEpochs', type=int, default=300)
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
    parser.add_argument('--dataset', default='./Gomez_T1', type=str, help='Dataset Path')
    parser.add_argument('--augment', action='store', type=int, default=0)
    parser.add_argument('--planes', action='store', type=str, default="sa,ch4,ch2")

    parser.add_argument('--weight-decay', '--wd', default=1e-8, type=float, metavar='W', help='weight decay (default: 1e-8)')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--save')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--opt', type=str, default='adam', choices=('sgd', 'adam', 'rmsprop'))
    args = parser.parse_args()
    planes = args.planes.split(",")
    best_prec1 = 100.
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.save = args.save or f'work/rotation_{args.planes}.base.{datestr()}'
    nll = True
    weight_decay = args.weight_decay

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    print("build vnet")
    model = VNetRegression(elu=False, nll=nll, outCH=len(planes))
    batch_size = args.batchSz

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        model.apply(weights_init)

    print('  + Number of params: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    if args.cuda:
        model = model.cuda()

    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    os.makedirs(args.save, exist_ok=True)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    print("loading training set")
    trainSet = GomezT1Rotation(root=args.dataset, portion=0.75, resolution=[128, 128, 128], augment=args.augment, planes=planes)
    trainLoader = DataLoader(trainSet, batch_size=batch_size, shuffle=True, **kwargs)
    print("loading test set")
    testSet = GomezT1Rotation(root=args.dataset,  portion=-0.25, resolution=[128, 128, 128], augment=args.augment, planes=planes)
    testLoader = DataLoader(testSet, batch_size=1, shuffle=False, **kwargs)

    if args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=1e-1, momentum=0.99, weight_decay=weight_decay)
    elif args.opt == 'adam':
        optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), weight_decay=weight_decay)

    trainF = open(os.path.join(args.save, 'train.csv'), 'w')
    testF = open(os.path.join(args.save, 'test.csv'), 'w')
    with open(os.path.join(args.save, 'dataset_split.txt'), 'w') as fp:
        fp.write(f"Train:{', '.join(trainSet.file_ids)}\nTest:{', '.join(testSet.file_ids)}")
    err_best = 100.
    for epoch in range(1, args.nEpochs + 1):
        adjust_opt(args.opt, optimizer, epoch)
        train(args, epoch, model, trainLoader, optimizer, trainF, planes)
        err = test(args, epoch, model, testLoader, optimizer, testF, planes)
        is_best = False
        if err < best_prec1:
            is_best = True
            best_prec1 = err
        save_checkpoint({'epoch': epoch, 'state_dict': model.state_dict(), 'best_prec1': best_prec1}, is_best, args.save, "rotation")

    trainF.close()
    testF.close()


def train(args, epoch, model, trainLoader, optimizer, trainF, planes):
    model.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    lossFunction = torch.nn.MSELoss()
    for batch_idx, (data, target) in enumerate(trainLoader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        if len(planes) == 1:
            target = target.view(target.numel())
        else:
            target = target.view(target.numel() // len(planes), len(planes))
        loss = lossFunction(output, target)
        # make_graph.save('/tmp/t.dot', loss.creator); assert(False)
        loss.backward()
        optimizer.step()
        nProcessed += len(data)
        # pred = output.data.max(1)[1]  # get the index of the max log-probability
        #incorrect = pred.ne(target.data).cpu().sum()
        #err = 100.*incorrect/target.numel()
        partialEpoch = epoch + batch_idx / len(trainLoader) - 1
        print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\t'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
            loss.item(),))

        trainF.write('{},{},{}\n'.format(partialEpoch, loss.item(), loss.item()))
        trainF.flush()


def test(args, epoch, model, testLoader, optimizer, testF, planes):
    model.eval()
    test_loss = 0
    lossFunction = torch.nn.MSELoss()
    for data, target in testLoader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        if len(planes) == 1:
            target = target.view(target.numel())
        else:
            target = target.view(target.numel() // len(planes), len(planes))
        output = model(data)
        test_loss += lossFunction(output, target).item()

    test_loss /= len(testLoader)  # loss function already averages over batch size
    print('Test set: Average loss: {:.4f}\n'.format(test_loss))

    testF.write('{},{}\n'.format(epoch, test_loss))
    testF.flush()
    return test_loss


def adjust_opt(optAlg, optimizer, epoch):
    if optAlg == 'sgd':
        if epoch < 5:
            lr = 1e-1
        elif epoch == 20:
            lr = 1e-2
        elif epoch == 30:
            lr = 1e-3
        else:
            return

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


if __name__ == '__main__':
    main()
