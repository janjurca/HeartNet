from typing import Tuple
import SimpleITK as sitk
import matplotlib.pylab as plt
import numpy as np
import matplotlib.lines as mlines
import math
import argparse
import glob
from matplotlib.widgets import Button
from scipy.interpolate import interp1d
import os
import json
from scipy.spatial.transform import Rotation as R
from sympy import Point3D
from sympy.geometry import Line3D
from matplotlib.widgets import RectangleSelector
import matplotlib.patches as patches

fig, (Side, Front) = None, (None, None)
selected_axis = None


class ItkImage:
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.augment_mhd_file()
        self.load()

    def augment_mhd_file(self):
        new_content = ""
        with open(self.filename, 'r') as fp:
            for line in fp.read().splitlines():
                if "TransformMatrix" in line:
                    line = "TransformMatrix = 1 0 0 0 1 0 0 0 1"
                elif "AnatomicalOrientation" in line:
                    line = "AnatomicalOrientation = LPS"
                new_content += line + "\n"
        with open(self.filename, 'w') as fp:
            fp.write(new_content)

    def load(self) -> None:
        self.image = sitk.ReadImage(self.filename, imageIO="MetaImageIO")  # TODO generalize for other formats

        original_CT = self.image
        dimension = original_CT.GetDimension()
        reference_physical_size = np.zeros(original_CT.GetDimension())
        reference_physical_size[:] = [(sz-1)*spc if sz*spc > mx else mx for sz, spc, mx in zip(original_CT.GetSize(), original_CT.GetSpacing(), reference_physical_size)]

        reference_origin = original_CT.GetOrigin()
        reference_direction = original_CT.GetDirection()

        reference_size = [128, 128, 128]
        reference_spacing = [phys_sz/(sz-1) for sz, phys_sz in zip(reference_size, reference_physical_size)]

        reference_image = sitk.Image(reference_size, original_CT.GetPixelIDValue())
        reference_image.SetOrigin(reference_origin)
        reference_image.SetSpacing(reference_spacing)
        reference_image.SetDirection(reference_direction)

        reference_center = np.array(reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize())/2.0))

        transform = sitk.AffineTransform(dimension)
        transform.SetMatrix(original_CT.GetDirection())

        transform.SetTranslation(np.array(original_CT.GetOrigin()) - reference_origin)

        centering_transform = sitk.TranslationTransform(dimension)
        img_center = np.array(original_CT.TransformContinuousIndexToPhysicalPoint(np.array(original_CT.GetSize())/2.0))
        centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
        centered_transform = sitk.Transform(transform)
        centered_transform = sitk.CompositeTransform([centered_transform])

        resampled_img = sitk.Resample(original_CT, reference_image, centered_transform, sitk.sitkLinear, 0.0)

        self.image = resampled_img
        self.image.SetSpacing([1, 1, 1])

        self.refresh()

    def refresh(self) -> None:
        self.ct_scan = sitk.GetArrayFromImage(self.image)
        self.origin = np.array(list(reversed(self.image.GetOrigin())))  # TODO handle different rotations
        self.spacing = np.array(list(reversed(self.image.GetSpacing())))

    def resolution(self) -> Tuple[int, int, int]:
        return (self.image.GetWidth(), self.image.GetHeight(), self.image.GetDepth())

    def resample(self, transform):
        """
        This function resamples (updates) an image using a specified transform
        :param image: The sitk image we are trying to transform
        :param transform: An sitk transform (ex. resizing, rotation, etc.
        :return: The transformed sitk image
        """
        interpolator = sitk.sitkBSpline
        default_value = 0
        return sitk.Resample(self.image, transform, interpolator, default_value)

    def get_center(self):
        """
        This function returns the physical center point of a 3d sitk image
        :param img: The sitk image we are trying to find the center of
        :return: The physical center point of the image
        """
        width, height, depth = self.image.GetSize()
        p = self.image.TransformIndexToPhysicalPoint((int(np.ceil(width/2)),
                                                     int(np.ceil(height/2)),
                                                     int(np.ceil(depth/2))))
        return p

    def rotation3d(self, theta_x, theta_y, theta_z):
        """
        This function rotates an image across each of the x, y, z axes by theta_x, theta_y, and theta_z degrees
        respectively
        :param image: An sitk MRI image
        :param theta_x: The amount of degrees the user wants the image rotated around the x axis
        :param theta_y: The amount of degrees the user wants the image rotated around the y axis
        :param theta_z: The amount of degrees the user wants the image rotated around the z axis
        :param show: Boolean, whether or not the user wants to see the result of the rotation
        :return: The rotated image
        """
        self.load()
        theta_x = np.deg2rad(theta_x)
        theta_y = np.deg2rad(theta_y)
        theta_z = np.deg2rad(theta_z)
        euler_transform = sitk.Euler3DTransform(self.get_center(), theta_x, theta_y, theta_z, (0, 0, 0))
        self.image = self.resample(euler_transform)

        self.refresh()


class VolumeImage:

    def eventSetup(self):
        def onScroll(event):
            if selected_axis is not self.ax:
                return

            if event.button == "up":
                self.index += 1
            if event.button == "down":
                self.index -= 1
            self.index = 0 if self.index < 0 else (len(self.image.ct_scan) - 1 if self.index > len(self.image.ct_scan) else self.index)
            self.ax.set_title(f"Slice: {self.index}")
            self.ax_data.set_data(self.image.ct_scan[self.index])
            self.fig.canvas.draw_idle()

        self.fig.canvas.mpl_connect('scroll_event', onScroll)

    def setIndex(self, index: int):
        self.index = index

    def redraw(self):
        self.ax_data.set_data(self.image.ct_scan[self.index])
        self.fig.canvas.draw_idle()


class PlotBoundingBox(VolumeImage):
    def __init__(self, image: ItkImage, ax, onSetBoundingBoxFunc=None, title="BoundingBox") -> None:
        self.image = image
        self.fig = fig
        self.ax = ax
        self.index = int(len(self.image.ct_scan)/2)
        self.boundingbox = None
        self.patch = None
        self.onSetBoundingBoxFunc = onSetBoundingBoxFunc
        self.ax_data = self.ax.imshow(self.image.ct_scan[self.index])
        self.eventSetup()

        def areaSelect(click, release):
            self.boundingbox = ((click.xdata, click.ydata), (release.xdata, release.ydata))
            if self.patch:
                self.patch.remove()
                self.patch = None
            self.patch = self.ax.add_patch(
                patches.Rectangle(
                    self.boundingbox[0],
                    self.boundingbox[1][0] - self.boundingbox[0][0],
                    self.boundingbox[1][1] - self.boundingbox[0][1],
                    linewidth=1, edgecolor='r', facecolor='none'))
            if self.onSetBoundingBoxFunc:
                self.onSetBoundingBoxFunc(self)

        self.rect_selector = RectangleSelector(self.ax, areaSelect, button=[1])


def ComputeBoundaries(plot: PlotBoundingBox):
    ((x1, y1), (x2, y2)) = plot.boundingbox
    if x1 > x2:
        x1, y1, x2, y2 = x2, y2, x1, y1

    res_x = plot.image.resolution()[0]
    res_y = plot.image.resolution()[1]

    mapper_init_x = interp1d([0, res_x], [0, 1])
    mapper_init_y = interp1d([0, res_y], [0, 1])

    x1, x2 = mapper_init_x([x1, x2])
    y1, y2 = mapper_init_y([y1, y2])

    return x1, y1, x2, y2


parser = argparse.ArgumentParser()
parser.add_argument('--input', action='store', default="Gomez_T1/a005/image.mhd",  help="Input files selection regex.")
parser.add_argument('--output', action='store', default="output/",  help="Input files selection regex.")
args = parser.parse_args()

top, bottom, left, right, front, back = None, None, None, None, None, None

for f in glob.glob(args.input):
    fig, (Side, Front) = plt.subplots(1, 2)

    target_dir = f'{args.output}/{"/".join(f.split(os.sep)[-3:-1])}'
    axnext = fig.add_axes([0.81, 0.05, 0.1, 0.075])

    def enter_axes(event):
        global selected_axis
        selected_axis = event.inaxes

    fig.canvas.mpl_connect('axes_enter_event', enter_axes)

    plotSide, plotFront = None, None,

    def onFrontSelected(plot: PlotBoundingBox):
        global top, bottom, left, right, front, back
        right, _, left, _ = ComputeBoundaries(plot)
        print("t:", top, "b:", bottom, "l:", left, "r:", right, "f:", front, "b:", back,)

    def onSideSelected(plot: PlotBoundingBox):
        global top, bottom, left, right, front, back
        front, top, back, bottom = ComputeBoundaries(plot)
        print("t:", top, "b:", bottom, "l:", left, "r:", right, "f:", front, "b:", back,)

    imageFront = ItkImage(f)
    imageFront.rotation3d(0, 90, 0)

    imageSide = ItkImage(f)

    plotFront = PlotBoundingBox(imageFront, Front, onSetBoundingBoxFunc=onFrontSelected)
    plotSide = PlotBoundingBox(imageSide, Side, onSetBoundingBoxFunc=onSideSelected)

    def nextFile(event):
        global target_dir, top, bottom, left, right, front, back

        os.makedirs(target_dir, exist_ok=True)

        with open(f"{target_dir}/position.json", 'w') as fp:
            data = {
                "top": top,
                "botom": bottom,
                "left": left,
                "right": right,
                "front": front,
                "back": back,
            }
            print(data)
            json.dump(data, fp)
            plt.close()

    bnext = Button(axnext, 'Save and next')
    bnext.on_clicked(nextFile)

    plt.show()
