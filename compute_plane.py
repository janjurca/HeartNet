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

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, required=True)
parser.add_argument('--image', type=str, required=True)
args = parser.parse_args()

image = ItkImage(args.image)

gt = ItkImage(args.file, image.res())


# sitk.Show(image.image)
sitk.Show(gt.image)

points = gt.points()

points = np.array(random.sample(points.tolist(), 500))

xs = points[:, 0]
ys = points[:, 1]
zs = points[:, 2]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(xs, ys, zs)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')


plane = Plane.best_fit(Points(points))

axis1 = Line([0, 0, 0], [0, image.res()[1], 0])
axis2 = Line([image.res()[0], 0, 0], [image.res()[0], image.res()[1], 0])
axis3 = Line([image.res()[0], 0, image.res()[2]], [image.res()[0], image.res()[1], image.res()[2]])
p1 = plane.intersect_line(axis1)
p2 = plane.intersect_line(axis2)
p3 = plane.intersect_line(axis3)


s_plane = sympy.Plane(sympy.Point3D(*p1), sympy.Point3D(*p2), sympy.Point3D(*p3))
zeroPoint = sympy.Point3D(0, 0, 0)
XAxis = Line3D(zeroPoint, sympy.Point3D(1, 0, 0))
YAxis = Line3D(zeroPoint, sympy.Point3D(0, 1, 0))
ZAxis = Line3D(zeroPoint, sympy.Point3D(0, 0, 1))

startPoint = sympy.Point3D(*[float(x) for x in s_plane.normal_vector]).unit
point2 = zeroPoint


lineX = Line2D(sympy.Point2D(point2.y, point2.z), sympy.Point2D(startPoint.y, startPoint.z))
lineY = Line2D(sympy.Point2D(point2.x, point2.z), sympy.Point2D(startPoint.x, startPoint.z))
lineZ = Line2D(sympy.Point2D(point2.x, point2.y), sympy.Point2D(startPoint.x, startPoint.y))

X_ANGLE = (math.degrees(float(lineX.angle_between(XAxis)))) % 90
Y_ANGLE = (math.degrees(float(lineY.angle_between(YAxis))))
Z_ANGLE = (math.degrees(float(lineZ.angle_between(ZAxis)))) - 90

print(X_ANGLE, Y_ANGLE, Z_ANGLE)

gt.rotation3d(Z_ANGLE, Y_ANGLE, X_ANGLE, reload=False, commit=True)
#gt.rotation3d(0, 90, 0, reload=False)

image.rotation3d(Z_ANGLE, Y_ANGLE, X_ANGLE, reload=False, commit=True)
#image.rotation3d(0, 90, 0, reload=False)

sitk.Show(gt.image)
# sitk.Show(image.image)


plot_3d(
    plane.plotter(alpha=0.2, lims_x=(-1000, 1000), lims_y=(-1000, 1000)),
    p1.plotter(),
    p2.plotter(),
    p3.plotter(),
)


plt.show()
