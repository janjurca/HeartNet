import SimpleITK as sitk
import matplotlib.pylab as plt
import numpy as np
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
from sympy import Point3D, Plane, Point2D
from sympy.geometry import Line3D, Line2D
from heartRotation.annotator.utils import PlotPlaneSelect, ItkImage
from heartRotation.annotator.gl import glvars, gllines
from matplotlib.widgets import Slider, Button, RadioButtons
from sympy import symbols
from mpl_toolkits.mplot3d import Axes3D
import pickle
from utils.compute_plane import rotateImage
from sympy.plotting import plot3d
from sympy.plotting import plot3d_parametric_line, plot3d_parametric_surface


SA_AXIS = None


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
    glvars.fig = plt.figure()
    glvars.fig.tight_layout()

    HLAax = glvars.fig.add_subplot(2, 4, 1)
    VLAax = glvars.fig.add_subplot(2, 4, 2)
    SAview = glvars.fig.add_subplot(2, 4, 3)
    CH4ax = glvars.fig.add_subplot(2, 4, 4)
    CH2ax = glvars.fig.add_subplot(2, 4, 5)
    GTax = glvars.fig.add_subplot(2, 4, 6)
    GTCH4ax = glvars.fig.add_subplot(2, 4, 7)
    GTCH2ax = glvars.fig.add_subplot(2, 4, 8)

    # PlaneView = glvars.fig.add_subplot(3, 3, 4, projection='3d')
    # PlaneView.set_xlabel('X')
    # PlaneView.set_ylabel('Z')
    # PlaneView.set_zlabel('Y')
    # PlaneView.axes.set_xlim3d(left=0, right=320)
    # PlaneView.axes.set_ylim3d(bottom=0, top=320)
    # PlaneView.axes.set_zlim3d(bottom=0, top=320)

    target_dir = f'{args.output}/{"/".join(f.split("/")[-3:-1])}'
    axnext = glvars.fig.add_axes([0.81, 0.05, 0.1, 0.075])
    axgenerate = glvars.fig.add_axes([0.6, 0.05, 0.1, 0.075])
    SA_AXIS = None

    def enter_axes(event):
        glvars.selected_axis = event.inaxes
    glvars.fig.canvas.mpl_connect('axes_enter_event', enter_axes)

    plotHLA, plotVLA, plotSA, plotGT, plotCH4, plotCH2, plotGTCH4, plotGTCH2 = None, None, None, None, None, None, None, None

    def on4CHSelected(plot: PlotPlaneSelect):
        ((x0, y0), (x1, y1)) = plotCH4.selectedLine
        line = Line2D(Point2D(x0, y0), Point2D(x1, y1))
        x_angle = math.degrees(float(line.angle_between(Line2D(Point2D(0, 0), Point2D(1, 0)))))
        plotCH2.image.rotation3d(0,  180-x_angle, 0)
        plotCH2.redraw()

    def showSAview():
        global SA_AXIS
        if not SA_AXIS:
            return
        zeroPoint = Point3D(0, 0, 0)
        Yplane = Plane(Point3D(0, glvars.height, 0), Point3D(1, glvars.height, 0), Point3D(0, glvars.height, 1))
        Ypoint = Yplane.intersection(SA_AXIS)[0]

        SAViewPlane = Plane(Ypoint, normal_vector=(SA_AXIS.points[1].x - SA_AXIS.points[0].x, SA_AXIS.points[1].y - SA_AXIS.points[0].y, SA_AXIS.points[1].z - SA_AXIS.points[0].z))

        XAxis = Line3D(zeroPoint, Point3D(1, 0, 0))
        YAxis = Line3D(zeroPoint, Point3D(0, 1, 0))
        ZAxis = Line3D(zeroPoint, Point3D(0, 0, 1))

        X_ANGLE = -(math.degrees(float(SAViewPlane.angle_between(XAxis))) - 90)
        Y_ANGLE = -(math.degrees(float(SAViewPlane.angle_between(YAxis))) - 90)
        Z_ANGLE = -(math.degrees(float(SAViewPlane.angle_between(ZAxis))) - 90)

        plotSA.image.rotation3d(0, Y_ANGLE,  -(180 - X_ANGLE))
        plotSA.setIndex(int(50))
        plotSA.redraw()

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
        SA_AXIS = Line3D((x0, y0, z0), (x1, y1, z1))
        showSAview()

        # SHOW 4CH
        ((x0, y0), (x1, y1)) = plotVLA.selectedLine
        line = Line2D(Point2D(x0, y0), Point2D(x1, y1))
        x_angle = math.degrees(float(line.angle_between(Line2D(Point2D(0, 0), Point2D(1, 0)))))
        plotCH4.image.load()
        for transform in plotHLA.image.transforms:
            plotCH4.image.applyTransform(transform, commit=False)
        plotCH4.image.rotation3d(x_angle, 0, 0, reload=False)

        plotCH4.redraw()

    imageHLA = ItkImage(f, name="PseudoHLA")
    imageHLA.rotation3d(0, 90, 270)

    plotHLA = PlotPlaneSelect(imageHLA, HLAax, onSetPlane=onHLASelected)
    plotVLA = PlotPlaneSelect(ItkImage(f, name="VLA"), VLAax, onSetPlane=onVLASelected)
    plotSA = PlotPlaneSelect(ItkImage(f, name="SAView"), SAview)
    plotCH4 = PlotPlaneSelect(ItkImage(f, name="CH4"), CH4ax, onSetPlane=on4CHSelected)
    plotCH2 = PlotPlaneSelect(ItkImage(f, name="CH2"), CH2ax)
    plotGT = PlotPlaneSelect(ItkImage(f, name="GTSA"), GTax)
    plotGTCH4 = PlotPlaneSelect(ItkImage(f, name="GTCH4"), GTCH4ax)
    plotGTCH2 = PlotPlaneSelect(ItkImage(f, name="GTCH2"), GTCH2ax)

    def computeValueByDistance(distance, length):
        mapper = interp1d([0, length], [0, 10])  # Tanh scale is defined here
        transformed_distance = mapper(distance)
        value = 1 - np.tanh(transformed_distance)
        return value

    def generateDistanceMap(index, shape):
        (depth, height, width) = shape
        gt = np.zeros((depth, height, width), dtype=float)
        for i in range(depth):
            gt[i:i+1, :, :] = computeValueByDistance(abs(index-i), depth)
        return gt

    def generate(event):
        # GTSA
        z = plotSA.index
        width, height, depth = plotSA.image.resolution()
        gtSA = generateDistanceMap(z, (depth, height, width))

        plotGT.image.setData(gtSA)
        plotGT.image.applyTransform(sitk.CompositeTransform(plotSA.image.transforms).GetInverse())
        plotGT.image.refresh()

        gtSA, indexgtSA = rotateImage(ItkImage(f, name="GTSA"), plotGT.image)
        plotGT.image = gtSA
        plotGT.setIndex(indexgtSA)
        plotGT.redraw()

        # GT4CH
        z = plotCH4.index
        width, height, depth = plotCH4.image.resolution()
        gt4CH = generateDistanceMap(z, (depth, height, width))
        plotGTCH4.image.setData(gt4CH)
        plotGTCH4.image.applyTransform(sitk.CompositeTransform(plotCH4.image.transforms).GetInverse())
        plotGTCH4.image.refresh()
        plotGTCH4.redraw()

        # GT4CH
        z = plotCH2.index
        width, height, depth = plotCH2.image.resolution()
        gt2CH = generateDistanceMap(z, (depth, height, width))
        plotGTCH2.image.setData(gt2CH)
        plotGTCH2.image.applyTransform(sitk.CompositeTransform(plotCH2.image.transforms).GetInverse())
        plotGTCH2.image.refresh()
        plotGTCH2.redraw()

    def nextFile(event):
        global target_dir
        os.makedirs(target_dir, exist_ok=True)
        generate(None)
        sitk.WriteImage(plotGT.image.image, target_dir + '/gtsa.mhd')
        sitk.WriteImage(plotGTCH4.image.image, target_dir + '/gtch4.mhd')
        sitk.WriteImage(plotGTCH2.image.image, target_dir + '/gtch2.mhd')

        #sitk.Show(plotSA.image.image, title="SA")
        #sitk.Show(plotGT.image.image, title="gt")

        plt.close()

    bnext = Button(axnext, 'Save and next')
    bnext.on_clicked(nextFile)
    bgenerate = Button(axgenerate, 'Generate')
    bgenerate.on_clicked(generate)

    plt.show()
