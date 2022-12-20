
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
import SimpleITK as sitk
import random


class GomezT1(Dataset):
    def __init__(self, root, portion=0.75, resolution=None, augment=False):
        self.data = []
        self.images = []
        self.resolution = resolution
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

                image.setHeartBox(
                    int(mapper_depth(loaded_meta["left"])), int(mapper_depth(loaded_meta["right"])),
                    int(mapper_height(loaded_meta["top"])), int(mapper_height(loaded_meta["botom"])),
                    int(mapper_width(loaded_meta["front"])), int(mapper_width(loaded_meta["back"])),
                )

                image.translate(30, 30, 30)
                gt[
                    int(mapper_depth(loaded_meta["left"])):int(mapper_depth(loaded_meta["right"])),
                    int(mapper_height(loaded_meta["top"])):int(mapper_height(loaded_meta["botom"])),
                    int(mapper_width(loaded_meta["front"])):int(mapper_width(loaded_meta["back"])),
                ] = 1

            self.data.append(
                (
                    torch.tensor([image.ct_scan]),
                    torch.tensor(gt, dtype=int)
                )
            )
            self.images.append(image)

        if augment:
            self.data.extend(self.augment())
        print("Dataset len: ", len(self.data))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def get(self, index):
        return self.data[index], self.images[index]

    def duplicateImage(self, image):
        im = ItkImage(image.filename, resolution=self.resolution)
        im.setHeartBox(
            left=image.heartBox["left"],
            right=image.heartBox["right"],
            top=image.heartBox["top"],
            bottom=image.heartBox["bottom"],
            front=image.heartBox["front"],
            back=image.heartBox["back"]
        )
        return im

    def generateVectors(self, image):
        width, height, depth = image.image.GetSize()
        gt = np.zeros((width, height, depth))
        gt[
            image.heartBox["left"]:image.heartBox["right"],
            image.heartBox["top"]:image.heartBox["bottom"],
            image.heartBox["front"]:image.heartBox["back"],
        ] = 1

        return (
            torch.tensor([image.ct_scan]),
            torch.tensor(gt, dtype=int)
        )

    def augment(self):
        augmented = []
        for i, image in enumerate(self.images):
            print(f"[{i}/{len(self.images)}]")
            xs = [0, -image.heartBox["front"], image.image.GetWidth()-image.heartBox["back"]]
            ys = [0, -image.heartBox["top"], image.image.GetHeight()-image.heartBox["bottom"]]
            zs = [0, -image.heartBox["left"], image.image.GetDepth()-image.heartBox["right"]]

            translations = []
            for x_move in xs:
                for y_move in ys:
                    for z_move in zs:
                        if x_move == 0 and y_move == 0 and z_move == 0:
                            continue
                        translations.append((x_move, y_move, z_move))

            #print("Translations len:", len(translations))
            for translation in translations:
                im = self.duplicateImage(image)

                im.translate(*translation)
                augmented.append(self.generateVectors(im))

        return augmented


class GomezT1Rotation(Dataset):
    def __init__(self, root, portion=0.75, resolution=None, augment=False):
        self.data = []
        self.images = []
        self.gtsas = []
        self.resolution = resolution
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
            gtsa = ItkImage(f"{root}/gt/{image_id}/gtsa.mhd", resolution=resolution)

            self.data.append(
                (
                    torch.tensor([image.ct_scan]),
                    torch.tensor(gtsa.ct_scan, dtype=torch.float32)
                )
            )
            self.images.append(image)
            self.gtsas.append(gtsa)

        if augment:
            self.data.extend(self.augment())
        print("Dataset len: ", len(self.data))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def get(self, index):
        return self.data[index], self.images[index], self.gtsas[index]

    def duplicateImage(self, image):
        im = ItkImage(image.filename, resolution=self.resolution)
        return im

    def generateVectors(self, image):
        pass

    def augment(self):
        augmented = []

        for i, (image, gtsa) in enumerate(zip(self.images, self.gtsas)):
            print(f"[{i}/{len(self.images)}]")
            for _ in range(10):
                theta_x = float(random.randrange(0, 2000))/100
                theta_y = float(random.randrange(0, 2000))/100
                theta_z = float(random.randrange(0, 2000))/100

                translate_x = float(random.randrange(0, 2000))/100
                translate_y = float(random.randrange(0, 2000))/100
                translate_z = float(random.randrange(0, 2000))/100

                im = self.duplicateImage(image)
                new_gtsa = self.duplicateImage(gtsa)

                im.translate(translate_x, translate_y, translate_z, reload=False, commit=False)
                im.rotation3d(theta_x, theta_y, theta_z, reload=False, commit=True)

                new_gtsa.translate(translate_x, translate_y, translate_z, reload=False, commit=False)
                new_gtsa.rotation3d(theta_x, theta_y, theta_z, reload=False, commit=True)

                augmented.append((
                    torch.tensor([im.ct_scan]),
                    torch.tensor(new_gtsa.ct_scan, dtype=torch.float32)
                ))

        return augmented
