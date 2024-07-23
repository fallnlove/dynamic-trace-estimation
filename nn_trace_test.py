import torch 
from torchvision import datasets, transforms
from data import * # get the dataset
from pyhessian import hessian # Hessian computation
from pytorchcv.model_provider import get_model as ptcv_get_model # model
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import expm, expm_multiply
from pylanczos import PyLanczos
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#*
# @file Different utility functions
# Copyright (c) Zhewei Yao, Amir Gholami
# All rights reserved.
# This file is part of PyHessian library.
#
# PyHessian is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyHessian is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyHessian.  If not, see <http://www.gnu.org/licenses/>.
#*

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


def getData(name='cifar10', train_bs=128, test_bs=1000):
    """
    Get the dataloader
    """
    if name == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        trainset = datasets.CIFAR10(root='../data',
                                    train=True,
                                    download=True,
                                    transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset,
                                                   batch_size=train_bs,
                                                   shuffle=True)

        testset = datasets.CIFAR10(root='../data',
                                   train=False,
                                   download=False,
                                   transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset,
                                                  batch_size=test_bs,
                                                  shuffle=False)
    if name == 'cifar10_without_dataaugmentation':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        trainset = datasets.CIFAR10(root='../data',
                                    train=True,
                                    download=True,
                                    transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset,
                                                   batch_size=train_bs,
                                                   shuffle=True)

        testset = datasets.CIFAR10(root='../data',
                                   train=False,
                                   download=False,
                                   transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset,
                                                  batch_size=test_bs,
                                                  shuffle=False)

    return train_loader, test_loader


def test(model, test_loader, cuda=True):
    """
    Get the test performance
    """
    model.eval()
    correct = 0
    total_num = 0
    for data, target in test_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        pred = output.data.max(
            1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
        total_num += len(data)
    print('testing_correct: ', correct / total_num, '\n')
    return correct / total_num

def hessian_vector_product(gradsH, params, v):
    """
    compute the hessian vector product of Hv, where
    gradsH is the gradient at the current point,
    params is the corresponding variables,
    v is the vector.
    """
    hv = torch.autograd.grad(gradsH,
                             params,
                             grad_outputs=v,
                             only_inputs=True,
                             retain_graph=True)
    return hv
def group_product(xs, ys):
    """
    the inner product of two lists of variables xs,ys
    :param xs:
    :param ys:
    :return:
    """
    return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])

def RademacherVector(n):
    """
        Input:
            n: num of components
        Output:
            vec: random vector from Rademacher distribution
    """

    v = [
        torch.randint_like(p, high=2)
        for p in n
    ]
    for v_i in v:
        v_i[v_i == 0] = -1

    return v

def Hutchinson(oracle, n, l: int):
    """
        Input:
            oracle: oracle for implicit matrix-vector multiplication with A
            n: size of the matrix A
            l: number of iteration to approximate the trace of the matrix A
        Output:
            approximation: approximation to the trace of A
    """

    assert l >= 0

    approximation = 0
    
    for iter in range(l):
        g = RademacherVector(n)
        approximation += group_product(oracle(g), g).cpu().item()
    
    return approximation / l

def SimpleHutchinson(oracles, n: int, l: int):
    """
        Input:
            oracles: oracles for implicit matrix-vector multiplication with A1, ..., Am
            n: matrix size
            l: number of iteration to approximate the trace of the matrixes A1, ..., Am
        Output:
            approximation: list of approximations to the trace of A1, ..., Am
    """
    assert len(oracles) > 0

    approximation = []
    
    for i in range(len(oracles)):
        approximation.append(Hutchinson(oracles[i], n[i], l))
    
    return approximation

def DeltaShiftRestart(oracles, n: int, l0: int, l: int, q: int):
    """
        DeltaShift algorithm, but restart every q iterations
        Input:
            oracles: oracles for implicit matrix-vector multiplication with A1, ..., Am
            n: matrix size
            l0: number of iteration to approximate the trace of the matrix A1
            l: number of iteration to approximate the trace of the other matrixes
            q: number of iterations to restart
        Output:
            approximation: list of approximations to the trace of A1, ..., Am
    """
    assert len(oracles) > 0

    approximation = []
    
    for i in range(len(oracles)):
        if i % q == 0:
            approximation.append(Hutchinson(oracles[i], n[i], l0))
        else:
            t = approximation[i - 1]
            for iter in range(l):
                g = RademacherVector(n[i])
                t += (group_product(oracles[i](g), g).cpu().item() - group_product(oracles[i - 1](g), g).cpu().item()) / l
            approximation.append(t)
    
    return approximation

def ParameterFreeDeltaShift(oracles, n: int, l0: int, l: int):
    """
        Parameter free version of DeltaShift algorithm, gamma estimates inplace
        
        Input:
            oracles: oracles for implicit matrix-vector multiplication with A1, ..., Am
            n: matrix size
            l0: number of iteration to approximate the trace of the matrix A1
            l: number of iteration to approximate the trace of the other matrixes
        Output:
            approximation: list of approximations to the trace of A1, ..., Am
    """
    assert len(oracles) > 0

    approximation = [0]
    N = 0

    for _ in range(l0):
        g = RademacherVector(n[0])
        z = oracles[0](g)
        approximation[0] += group_product(z, g).cpu().item()
        N += group_product(z, z).cpu().item()

    approximation[0] /= l0
    N /= l0
    variance = 2 / l0 * N

    l //= 2
    
    for i in range(1, len(oracles)):
        z = []
        w = []
        g = []
        for j in range(l):
            cur = RademacherVector(n[i])
            z.append(oracles[i - 1](cur))
            w.append(oracles[i](cur))
            g.append(cur)

        N = 0
        M = 0
        C = 0
        for j in range(l):
            N += group_product(z[j], z[j]).cpu().item() / l
            M += group_product(w[j], w[j]).cpu().item() / l
            C += group_product(w[j], z[j]).cpu().item() / l

        gamma = 1 - (2 * C) / (l * variance + 2 * N)

        t = (1 - gamma) * approximation[i - 1]
        for j in range(l):
            t += (group_product(g[j], w[j]).cpu().item() - (1 - gamma) * group_product(g[j], z[j]).cpu().item()) / l

        approximation.append(t)
        variance = (1 - gamma)**2 * variance + 2 / l * (M + (1 - gamma)**2 * N - 2 * (1 - gamma) * C)
    
    return approximation

def write_array(arr, file_name):
    f = open(file_name, 'w')
    for i in arr:
        f.write(str(i) + " ")
    f.write("\n")
    f.close()

def CheckAlgorithms(oracles, n, correct_ans, title = None):
    """
        Generate graphic with relative error
        Input:
            oracles: list of oracle to compute matrix-vector multiplication
            n: matrix size
            correct_ans: list of the trace of the given matrixes
            title: title of the graphic
        Output:
            -
    """

    l0 = 100
    l = 50

    correct_ans = np.array(correct_ans)

    simpl_hutchinson = np.array(SimpleHutchinson(oracles, n, l))
    write_array(abs(correct_ans - simpl_hutchinson) / max(correct_ans), 'Hutchinson.txt')

    print("1")

    delta_shift = np.array(ParameterFreeDeltaShift(oracles, n, l0, l - 50 // (len(oracles) - 1)))
    write_array(abs(correct_ans - delta_shift) / max(correct_ans), 'Deltshift.txt')

    print("2")

    delta_shift_r = np.array(DeltaShiftRestart(oracles, n, l0, l - 50 // 19, 20))
    write_array(abs(correct_ans - delta_shift_r) / max(correct_ans), 'Deltshiftrest.txt')

# get the model 
import datetime


model = ptcv_get_model("resnet20_cifar10", pretrained=True)
# change the model to eval mode to disable running stats upate
model.eval()

# create loss function
criterion = torch.nn.CrossEntropyLoss()

# get dataset 
train_loader, test_loader = getData()

computable = []

oracles = []
correct_ans = []
n = []

for ab, i in zip(train_loader, range(len(train_loader))):
    inputs, targets = ab
    model = model.cuda()
    inputs, targets = inputs.cuda(), targets.cuda()
    hessian_comp = hessian(model, criterion, data=(inputs, targets), cuda=False)

    print(datetime.datetime.now())
    correct_ans.append(Hutchinson((lambda x: hessian_vector_product(hessian_comp.gradsH, hessian_comp.params, x)), hessian_comp.params, 500))
    print(datetime.datetime.now())

print(correct_ans)
write_array(correct_ans, "correct.txt")
