
import pickle
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.utils.data
import random
from utils.itkImage import ItkImage
import glob
import os
import json
from sympy import Point3D
from sympy.geometry import Line3D, Segment3D
import math
from scipy.interpolate import interp1d


class GomezT1(Dataset):
    def __init__(self, root, portion=0.75, resolution=None):
        self.data = []
        self.images = []
        files = glob.glob(f"{root}/original/*/image.mhd")
        if portion > 0:
            files = files[:int(portion*len(files))]
        else:
            files = files[int((1+portion)*len(files)):]
        for file in files:
            image = ItkImage(file, resolution=resolution)
            width, height, depth = image.image.GetSize()
            image_id = file.split(os.sep)[-2]
            gt = np.zeros((width, height, depth))

            with open(f"{root}/positions/{image_id}/position.json") as fp:
                loaded_meta = json.load(fp)

                mapper_width = interp1d([0, 1], [0, width])
                mapper_height = interp1d([0, 1], [0, height])
                mapper_depth = interp1d([1, 0], [0, depth])

                gt[
                    int(mapper_depth(loaded_meta["left"])):int(mapper_depth(loaded_meta["right"])),
                    int(mapper_height(loaded_meta["top"])):int(mapper_height(loaded_meta["botom"])),
                    int(mapper_width(loaded_meta["front"])):int(mapper_width(loaded_meta["back"])),
                ] = 1
                # {"t": 0.24149697580645135, "b": 0.6303679435483871, "l": 0.9960937499999998, "r": 0.18440020161290316, "f": 0.3065776209677419}

            self.data.append(
                (
                    torch.tensor([image.ct_scan]),
                    torch.tensor(gt, dtype=int)
                )
            )
            self.images.append(image.image)

        print("Dataset len: ", len(self.data))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def get(self, index):
        return self.data[index], self.images[index]
