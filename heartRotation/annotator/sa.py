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
from sympy import Point3D, Plane
from sympy.geometry import Line3D
from utils import PlotPlaneSelect, ItkImage
from gl import glvars, gllines
from matplotlib.widgets import Slider, Button, RadioButtons
from sympy import symbols
from mpl_toolkits.mplot3d import Axes3D
import pickle

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
    glvars.fig.set_figheight(15)
    glvars.fig.set_figwidth(15)

    HLAax = glvars.fig.add_subplot(1, 4, 1)
    VLAax = glvars.fig.add_subplot(1, 4, 2)
    SAview = glvars.fig.add_subplot(1, 4, 3)
    Originalax = glvars.fig.add_subplot(1, 4, 4)
    #PlaneView = glvars.fig.add_subplot(3, 3, 4, projection='3d')
    # PlaneView.set_xlabel('X')
    # PlaneView.set_ylabel('Z')
    # PlaneView.set_zlabel('Y')
    #PlaneView.axes.set_xlim3d(left=0, right=320)
    #PlaneView.axes.set_ylim3d(bottom=0, top=320)
    #PlaneView.axes.set_zlim3d(bottom=0, top=320)

    target_dir = f'{args.output}/{"/".join(f.split("/")[-3:-1])}'
    axnext = glvars.fig.add_axes([0.81, 0.05, 0.1, 0.075])
    SA_AXIS = None

    axHeight = glvars.fig.add_axes([0.25, 0.1, 0.4, 0.03])

    def enter_axes(event):
        glvars.selected_axis = event.inaxes
    glvars.fig.canvas.mpl_connect('axes_enter_event', enter_axes)

    plotHLA, plotVLA, plotSAview = None, None, None

    def showSAview():
        global SA_AXIS
        if not SA_AXIS:
            return
        zeroPoint = Point3D(0, 0, 0)
        Yplane = Plane(Point3D(0, glvars.height, 0), Point3D(1, glvars.height, 0), Point3D(0, glvars.height, 1))
        Ypoint = Yplane.intersection(SA_AXIS)[0]

        print(Ypoint)
        SAViewPlane = Plane(Ypoint, normal_vector=(SA_AXIS.points[1].x - SA_AXIS.points[0].x, SA_AXIS.points[1].y - SA_AXIS.points[0].y, SA_AXIS.points[1].z - SA_AXIS.points[0].z))

        point = np.array([float(Ypoint.x), float(Ypoint.y), float(Ypoint.z)])
        normal = np.array([float(SA_AXIS.points[1].x), float(SA_AXIS.points[1].y), float(SA_AXIS.points[1].z)])
        print(point, normal)
        # a plane is a*x+b*y+c*z+d=0
        # [a,b,c] is the normal. Thus, we have to calculate
        # d and we're set
        d = -point.dot(normal)

        # create x,y
        xx, yy = np.meshgrid(range(200), range(200))
        # calculate corresponding z
        z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]
        # plot the surface
        '''
        PlaneView.plot_surface(xx, yy, z)

        x, y, z = [float(Ypoint.x), float(SA_AXIS.points[1].x)], [float(Ypoint.y), float(SA_AXIS.points[1].y)], [float(Ypoint.z), float(SA_AXIS.points[1].z)]
        PlaneView.scatter(x, y, z, c='red', s=100)
        PlaneView.plot(x, y, z, color='black')
        '''
        XAxis = Line3D(zeroPoint, Point3D(1, 0, 0))
        YAxis = Line3D(zeroPoint, Point3D(0, 1, 0))
        ZAxis = Line3D(zeroPoint, Point3D(0, 0, 1))

        X_ANGLE = -(math.degrees(float(SAViewPlane.angle_between(XAxis))) - 90)
        Y_ANGLE = -(math.degrees(float(SAViewPlane.angle_between(YAxis))) - 90)
        Z_ANGLE = -(math.degrees(float(SAViewPlane.angle_between(ZAxis))) - 90)

        print("PLANE_ANGLES", X_ANGLE-90, Y_ANGLE-90, Z_ANGLE-90)
        print("Center Point", float(Ypoint.x), float(Ypoint.y), float(Ypoint.z))
        print("SA AXIS A:", float(SA_AXIS.points[0].x), float(SA_AXIS.points[0].y), float(SA_AXIS.points[0].z))
        print("SA AXIS B:", float(SA_AXIS.points[1].x), float(SA_AXIS.points[1].y), float(SA_AXIS.points[1].z))

        print("Normal", [float(x) for x in [SA_AXIS.points[1].x - SA_AXIS.points[0].x, SA_AXIS.points[1].y - SA_AXIS.points[0].y, SA_AXIS.points[1].z - SA_AXIS.points[0].z]])

        plotSAview.image.rotation3d(0, Y_ANGLE,  -(180 - X_ANGLE))
        plotSAview.setIndex(int(glvars.height*100))
        plotSAview.redraw()

        glvars.center_point = Ypoint
        glvars.plane = SAViewPlane
        glvars.SA_AXIS = SA_AXIS

    def onHLASelected(plot: PlotPlaneSelect):
        plotVLA.image.rotation3d(0,  180-ComputeLineAngle(plot), 0)
        plotVLA.redraw()
        plotVLAview.image.rotation3d(0,  180-ComputeLineAngle(plot), 0)
        plotVLAview.redraw()

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
        SA_AXIS = Line3D((x0, y0, z0), (x1, y1, z1))
        showSAview()

    imageHLA = ItkImage(f, name="PseudoHLA")
    imageHLA.rotation3d(0, 90, 270)

    plotHLA = PlotPlaneSelect(imageHLA, HLAax, onSetPlane=onHLASelected)
    plotVLA = PlotPlaneSelect(ItkImage(f, name="VLA"), VLAax, onSetPlane=onVLASelected)
    plotSAview = PlotPlaneSelect(ItkImage(f, name="SAView"), SAview, onSetPlane=onSASelected)
    PlotPlaneSelect(ItkImage(f, name="Original"), Originalax)

    def nextFile(event):
        global target_dir

        os.makedirs(target_dir, exist_ok=True)

        if not glvars.SA_AXIS or not glvars.plane or not glvars.center_point:
            print("SA AXIS not set. DO IT!")
            return
        with open(f"{target_dir}/sa.p", 'wb') as fp:
            data = {
                "plane": glvars.plane,
                "axis": glvars.SA_AXIS,
                "point": glvars.center_point
            }
            pickle.dump(data, fp)
            plt.close()

    bnext = Button(axnext, 'Save and next')
    bnext.on_clicked(nextFile)

    freq_slider = Slider(
        ax=axHeight,
        label='Height',
        valmin=0,
        valmax=1,
        valinit=0.5,
    )

    def update(x):
        glvars.height = x
        showSAview()
    freq_slider.on_changed(update)

    plt.show()
