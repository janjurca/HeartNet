import pickle
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.utils.data
import random
from itkImage import ItkImage
import glob
import os
import json
import simpy
from sympy import Point3D
from sympy.geometry import Line3D, Segment3D
import torch

from torch.utils.data import DataLoader
import os

import shutil
import vnet
from datasets import GomezT1
import vnetone
import sys

root = sys.argv[1]
image_id = sys.argv[2]

files = glob.glob(f"{root}/original/{image_id}/image.mhd")

image = ItkImage(file)
width, height, depth = image.resolution()
with open(f"{root}/annotated/{image_id}/meta.json") as fp:
    loaded_meta = json.load(fp)
    ret = torch.zeros(depth, width, height).tolist()
    line = Line3D(
        Point3D(loaded_meta["SA"]["A"]["x"], loaded_meta["SA"]["A"]["y"], loaded_meta["SA"]["A"]["z"]),
        Point3D(loaded_meta["SA"]["B"]["x"], loaded_meta["SA"]["B"]["y"], loaded_meta["SA"]["B"]["z"]),
    )
    print("Generating GT", image_id)
    for x in range(width):  # VERY VERY slow implementation of generating GT
        for y in range(height):
            for z in range(depth):
                ret[z][x][y] = line.distance(Point3D(x, y, z))  # IS WIDTH WIDTH and HEIGHT HEIGT (is indexing ok?)
with open(f"{root}/annotated/{image_id}/GT.pickle", 'w') as fp:
    pickle.dump(ret, fp)
