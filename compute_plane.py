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
    points = np.array(random.sample(image.points().tolist(), 5000))
    xs, ys, zs = points[:, 0], points[:, 1], points[:, 2]
    return Plane.best_fit(Points(points)), (xs, ys, zs)


def planeAngles(plane):
    xangle = math.degrees(Vector([1, 0, 0]).angle_between(plane.normal))
    yangle = math.degrees(Vector([0, 1, 0]).angle_between(plane.normal))
    zangle = math.degrees(Vector([0, 0, 1]).angle_between(plane.normal))
    print(plane.point)
    print("V 1:", math.degrees(Vector([1, 0]).angle_between(Vector(plane.normal[:2]))))  # XY angle
    print("V 2:", math.degrees(Vector([0, 1]).angle_between(Vector(plane.normal[:2]))))
    print("V 3:", math.degrees(Vector([1, 0]).angle_between(Vector(plane.normal[1:]))))
    print("V 4:", math.degrees(Vector([0, 1]).angle_between(Vector(plane.normal[1:]))))
    xangle = 90 - math.degrees(Vector([1, 0]).angle_between(plane.normal[1:]))
    print(xangle, yangle, zangle)
    return xangle, yangle, zangle


parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, required=True)
parser.add_argument('--image', type=str, required=True)
args = parser.parse_args()

image = ItkImage(args.image)
gt = ItkImage(args.file, image.res())
# sitk.Show(image.image)
sitk.Show(gt.image)


plane, points = fitPlane(gt)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(*points)


X_ANGLE, Y_ANGLE, Z_ANGLE = planeAngles(plane)
print(X_ANGLE, Y_ANGLE, Z_ANGLE)


gt.rotation3d(0, 0, X_ANGLE, reload=False, commit=True)
#gt.transformByMatrix(matrix, reload=False, commit=True)
# gt.rotation3d(0, 90, 0, reload=False)

#image.rotation3d(Z_ANGLE, Y_ANGLE, X_ANGLE, reload=False, commit=True)
# image.rotation3d(0, 90, 0, reload=False)

sitk.Show(gt.image)
# sitk.Show(image.image)


plot_3d(
    plane.plotter(alpha=0.2, lims_x=(-1000, 1000), lims_y=(-1000, 1000)),
)


plt.show()
