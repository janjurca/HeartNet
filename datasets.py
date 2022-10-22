
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
import math
from scipy.interpolate import interp1d


class GomezT1(Dataset):
    def __init__(self, root, portion=0.75):
        self.data = []
        files = glob.glob(f"{root}/original/*/image.mhd")
        if portion > 0:
            files = files[:int(portion*len(files))]
        else:
            files = files[int((1+portion)*len(files)):]
        for file in files:
            image = ItkImage(file)
            width, height, depth = image.resolution()
            image_id = file.split(os.sep)[-2]

            with open(f"{root}/annotated/{image_id}/meta.json") as fp:
                loaded_meta = json.load(fp)

                zeroPoint = Point3D(0, 0, 0)
                SA_AXIS = Line3D(
                    Point3D(loaded_meta["SA"]["A"]["x"], loaded_meta["SA"]["A"]["y"], loaded_meta["SA"]["A"]["z"]),
                    Point3D(loaded_meta["SA"]["B"]["x"], loaded_meta["SA"]["B"]["y"], loaded_meta["SA"]["B"]["z"]),
                )
                X_ANGLE = math.degrees(float(SA_AXIS.angle_between(Line3D(zeroPoint, Point3D(1, 0, 0)))))
                Y_ANGLE = math.degrees(float(SA_AXIS.angle_between(Line3D(zeroPoint, Point3D(0, 1, 0)))))
                Z_ANGLE = math.degrees(float(SA_AXIS.angle_between(Line3D(zeroPoint, Point3D(0, 0, 1)))))
                m = interp1d([0, 360], [0, 1])
            self.data.append(
                (
                    torch.tensor([image.ct_scan[:90, :, :]]),
                    torch.tensor([float(m(X_ANGLE)), float(m(Y_ANGLE)), float(m(Z_ANGLE))]),
                )
            )
        print("Dataset len: ", len(self.data))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    # def generateGT(self, root, image_id, image):
    #    width, height, depth = image.resolution()
    #    loaded_meta = None
    #    with open(f"{root}/annotated/{image_id}/meta.json") as fp:
    #        loaded_meta = json.load(fp)
    #        ret = torch.zeros(depth, width, height).tolist()
    #        line = Line3D(
    #            Point3D(loaded_meta["SA"]["A"]["x"], loaded_meta["SA"]["A"]["y"], loaded_meta["SA"]["A"]["z"]),
    #            Point3D(loaded_meta["SA"]["B"]["x"], loaded_meta["SA"]["B"]["y"], loaded_meta["SA"]["B"]["z"]),
    #        )
    #        print("Generating GT", image_id)
    #        for x in range(width):  # VERY VERY slow implementation of generating GT
    #            for y in range(height):
    #                print(x, y)
    #                for z in range(depth):
    #                    ret[z][x][y] = line.distance(Point3D(x, y, z))  # IS WIDTH WIDTH and HEIGHT HEIGT (is indexing ok?)
    #    with open(f"{root}/annotated/{image_id}/GT.pickle", 'w') as fp:
    #        pickle.dump(ret, fp)
    #    return ret
