import matplotlib.pyplot as plt
import numpy as np
from utils.itkImage import ItkImage
import argparse
from skspatial.objects import Plane, Points
from skspatial.plotting import plot_3d
from skspatial.objects import Line, Plane
import sympy
from sympy.geometry import Line3D, Line2D
import math
import SimpleITK as sitk
import random
from math import sqrt
from skspatial.objects import Vector


def fitPlane(image: ItkImage):
    points = np.array(random.sample(image.points().tolist(), 5000 if len(image.points().tolist()) > 5000 else len(image.points().tolist())))
    xs, ys, zs = points[:, 0], points[:, 1], points[:, 2]
    return Plane.best_fit(Points(points)), (xs, ys, zs)


def planeAngles(plane):
    xangle = 90 - math.degrees(Vector([1, 0]).angle_between(plane.normal[1:]))
    yangle = 90 - math.degrees(Vector([0, 1]).angle_between(Vector(plane.normal[:2])))
    zangle = 90 - math.degrees(Vector([1, 0]).angle_between(Vector([plane.normal[0], plane.normal[2]])))
    return xangle, yangle, zangle


def comparePlanes(mask1, mask2):
    plane1, points = fitPlane(mask1)
    plane2, points = fitPlane(mask2)

    angle = plane1.normal.angle_between(plane2.normal)
    angle = math.degrees(angle)
    return angle


def rotateImage(image, mask):
    plane, points = fitPlane(mask)
    X_ANGLE, Y_ANGLE, Z_ANGLE = planeAngles(plane)
    mask.rotation3d(0, 0, X_ANGLE, reload=False, commit=True)
    image.rotation3d(0, 0, X_ANGLE, reload=False, commit=True)
    plane, points = fitPlane(mask)
    X_ANGLE, Y_ANGLE, Z_ANGLE = planeAngles(plane)
    mask.rotation3d(0, 90-Z_ANGLE, 0, reload=False, commit=True)
    image.rotation3d(0, 90-Z_ANGLE, 0, reload=False, commit=True)
    index = np.argmax(np.sum(np.sum(mask.ct_scan, 1), 1))
    return image, index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask', type=str, required=True)
    parser.add_argument('--image', type=str, required=True)
    args = parser.parse_args()

    image = ItkImage(args.image)
    mask = ItkImage(args.mask)

    image, index = rotateImage(image, mask)
    # sitk.Show(image.image)

    # plot_3d(
    #    plane.plotter(alpha=0.2, lims_x=(-1000, 1000), lims_y=(-1000, 1000)),
    # )
    # plt.show()
