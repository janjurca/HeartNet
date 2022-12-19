from fileinput import filename
from time import sleep
from typing import Tuple
import SimpleITK as sitk
import matplotlib.pylab as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib.widgets import RectangleSelector
import matplotlib.lines as mlines
import math
import argparse
import glob
from matplotlib.widgets import Button
from scipy.interpolate import interp1d
import os
import json
from scipy.spatial.transform import Rotation as R
from sympy import Point, Line
from sympy.solvers import solve
from sympy import Symbol
from sympy import Point3D
from sympy.geometry import Line3D
from utils import PlotPlaneSelect, ItkImage
from gl import glvars


def ComputeLineAngle(plot: PlotPlaneSelect):
    ((x1, y1), (x2, y2)) = plot.selectedLine
    if x1 > x2:
        x1, y1, x2, y2 = x2, y2, x1, y1
    quadrant = -180 if y1 < y2 else 0

    a = abs(plot.selectedLine[0][0] - plot.selectedLine[1][0])
    b = abs(plot.selectedLine[0][1] - plot.selectedLine[1][1])
    rad = math.atan(a/b)
    angle = abs(math.degrees(rad) + quadrant)
    print(f'atan(x) :{rad}, deg: {angle}, q: {quadrant}')
    return angle


parser = argparse.ArgumentParser()
parser.add_argument('--input', action='store', default="Gomez_T1/a001/image.mhd",  help="Input files selection regex.")
parser.add_argument('--output', action='store', default="output/",  help="Input files selection regex.")
args = parser.parse_args()


for f in glob.glob(args.input):
    glvars.fig, (HLAax, VLAax, SAax) = plt.subplots(1, 3)

    target_dir = f'{args.output}/{"/".join(f.split("/")[-3:-1])}'
    axnext = glvars.fig.add_axes([0.81, 0.05, 0.1, 0.075])
    SA_AXIS = None

    def enter_axes(event):
        glvars.selected_axis = event.inaxes

    glvars.fig.canvas.mpl_connect('axes_enter_event', enter_axes)

    plotHLA, plotVLA, plotSA = None, None, None

    def onHLASelected(plot: PlotPlaneSelect):
        plotVLA.image.rotation3d(0,  180-ComputeLineAngle(plot), 0)
        plotVLA.redraw()

    def onVLASelected(plot: PlotPlaneSelect):
        global SA_AXIS
        ((x0, y0), (x1, y1)) = plotHLA.selectedLine
        ((x2, y2), (x3, y3)) = plotVLA.selectedLine
        res = plotHLA.image.resolution()[0]
        mapper = interp1d([0, res], [0, 1])
        mapper_rev = interp1d([0, 1], [0, plotHLA.image.resolution()[0]])
        mapper100_rev = interp1d([0, 1], [0, plotHLA.image.resolution()[2]])
        x0, y0, x1, y1, x2, y2, x3, y3 = mapper([x0, y0, x1, y1, x2, y2, x3, y3])

        VLA_line = Line(Point(x2, y2), Point(x3, y3))
        (a, b, c) = VLA_line.coefficients
        y = Symbol('y')
        z0 = 1 - float(solve(a*(1-y0) + b*y + c, y)[0])
        z1 = 1 - float(solve(a*(1-y1) + b*y + c, y)[0])
        print()
        print("Before:", (x0, y0, z0), (x1, y1, z1))
        print("Before coords:", mapper_rev((x0, y0)), mapper100_rev(z0), mapper_rev((x1, y1)), mapper100_rev(z1))
        x0, y0, z0, x1, y1, z1 = x0 - 0.5, y0 - 0.5, z0 - 0.5, x1 - 0.5, y1 - 0.5, z1 - 0.5
        print("Sub:", (x0, y0, z0), (x1, y1, z1))
        r = R.from_euler('xyz', [0, 90, 270], degrees=True)
        x0, y0, z0 = r.apply(np.array([x0, y0, z0]))
        x1, y1, z1 = r.apply(np.array([x1, y1, z1]))
        print("Sub2:", (x0, y0, z0), (x1, y1, z1))
        x0, y0, z0, x1, y1, z1 = x0 + 0.5, y0 + 0.5, z0 + 0.5, x1 + 0.5, y1 + 0.5, z1 + 0.5
        print("After:", (x0, y0, z0), (x1, y1, z1))

        x0, y0, x1, y1 = mapper_rev([x0, y0, x1, y1])
        z0, z1 = mapper100_rev([z0, z1])

        print("After coords:", (x0, y0, z0), (x1, y1, z1))

        zeroPoint = Point3D(0, 0, 0)
        XAxis = Line3D(zeroPoint, Point3D(1, 0, 0))
        YAxis = Line3D(zeroPoint, Point3D(0, 1, 0))
        ZAxis = Line3D(zeroPoint, Point3D(0, 0, 1))
        SA_AXIS = Line3D((x0, y0, z0), (x1, y1, z1))
        X_ANGLE = math.degrees(float(SA_AXIS.angle_between(XAxis)))
        Y_ANGLE = math.degrees(float(SA_AXIS.angle_between(YAxis)))
        Z_ANGLE = math.degrees(float(SA_AXIS.angle_between(ZAxis)))
        print(X_ANGLE, Y_ANGLE, Z_ANGLE)
        plotSA.image.rotation3d(0, Y_ANGLE,  -(180 - X_ANGLE))
        plotSA.redraw()

    imageHLA = ItkImage(f)
    imageHLA.rotation3d(0, 90, 270)

    plotHLA = PlotPlaneSelect(imageHLA, HLAax, onSetPlane=onHLASelected)
    plotVLA = PlotPlaneSelect(ItkImage(f), VLAax, onSetPlane=onVLASelected)
    plotSA = PlotPlaneSelect(ItkImage(f), SAax)

    def nextFile(event):
        global SA_AXIS, target_dir

        os.makedirs(target_dir, exist_ok=True)

        if not SA_AXIS:
            print("SA AXIS not set. DO IT!")
            return
        with open(f"{target_dir}/meta.json", 'w') as fp:
            data = {
                "SA": {
                    "A": {
                        "x": float(SA_AXIS.points[0].x),
                        "y": float(SA_AXIS.points[0].y),
                        "z": float(SA_AXIS.points[0].z)
                    },
                    "B": {
                        "x": float(SA_AXIS.points[1].x),
                        "y": float(SA_AXIS.points[1].y),
                        "z": float(SA_AXIS.points[1].z),
                    },
                },
            }
            print(data)
            json.dump(data, fp)
            img = sitk.GetImageFromArray(plotSA.image.ct_scan)
            img.SetOrigin(plotSA.image.origin)
            img.SetSpacing(plotSA.image.spacing)

            sitk.WriteImage(img, target_dir + '/image.mhd')
            plt.close()

    bnext = Button(axnext, 'Save and next')
    bnext.on_clicked(nextFile)

    plt.show()
