import SimpleITK as sitk
import numpy as np
import matplotlib.lines as mlines
from sympy import Point3D, Point2D
from heartRotation.annotator.gl import glvars, gllines


class ItkImage:
    def __init__(self, filename: str, name) -> None:
        self.name = name
        self.filename = filename
        self.augment_mhd_file()
        self.load()
        self.transforms = []
        self.todo_transforms = []

    def clone(self):
        im = ItkImage(self.filename, self.name + "clone")
        for transform in self.transforms:
            im.applyTransform(transform)

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
        gllines.clearRotation(self.name)
        self.transforms = []
        self.refresh()

    def refresh(self) -> None:
        # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
        self.ct_scan = sitk.GetArrayFromImage(self.image)
        # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
        self.origin = np.array(list(reversed(self.image.GetOrigin())))  # TODO handle different rotations
        # Read the spacing along each dimension
        self.spacing = np.array(list(reversed(self.image.GetSpacing())))

    def resolution(self):
        return (self.image.GetWidth(), self.image.GetHeight(), self.image.GetDepth())

    def resample(self, transform):
        """
        This function resamples (updates) an image using a specified transform
        :param image: The sitk image we are trying to transform
        :param transform: An sitk transform (ex. resizing, rotation, etc.
        :return: The transformed sitk image
        """
        reference_image = self.image
        interpolator = sitk.sitkBSpline
        default_value = 0
        return sitk.Resample(self.image, reference_image, transform,   interpolator, default_value, useNearestNeighborExtrapolator=True)

    def points(self, threshold=0.5):
        print(np.max(self.ct_scan))
        return np.argwhere(self.ct_scan > threshold)

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
        print("center", p)
        return p

    def rotation3d(self, theta_x, theta_y, theta_z, reload=True, commit=True):
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
        if reload:
            self.load()
        theta_x = np.deg2rad(theta_x)
        theta_y = np.deg2rad(theta_y)
        theta_z = np.deg2rad(theta_z)
        euler_transform = sitk.Euler3DTransform(self.get_center(), theta_x, theta_y, theta_z, (0, 0, 0))
        self.applyTransform(euler_transform, commit)

    def applyTransform(self, transform, commit=True):
        self.todo_transforms.append(transform)
        if commit:
            self.commitTransform()
            self.refresh()

    def commitTransform(self):
        self.image = self.resample(sitk.CompositeTransform(self.todo_transforms))
        self.transforms.extend(self.todo_transforms)

        self.todo_transforms = []

    def setData(self, data):
        im = sitk.GetImageFromArray(data)
        im.SetDirection(self.image.GetDirection())
        im.SetOrigin(self.image.GetOrigin())
        im.SetSpacing(self.image.GetSpacing())
        self.image = im
        self.refresh()


class VolumeImage:

    def eventSetup(self):
        def onScroll(event):
            if glvars.selected_axis is not self.ax:
                return

            if event.button == "up":
                self.index += 1
            if event.button == "down":
                self.index -= 1
            self.index = 0 if self.index < 0 else (len(self.image.ct_scan) - 1 if self.index > len(self.image.ct_scan) else self.index)
            self.redraw()
            if self.onScroll:
                self.onScroll(self)

        glvars.fig.canvas.mpl_connect('scroll_event', onScroll)

    def setIndex(self, index: int):
        self.index = index

    def redraw(self):
        self.ax.set_title(f"{self.title} | {self.index}")
        self.ax_data.set_data(self.image.ct_scan[self.index])
        glvars.fig.canvas.draw_idle()


class PlotPlaneSelect(VolumeImage):
    def __init__(self, image: ItkImage, ax, onSetPlane=None, onScroll=None) -> None:
        self.onScroll = onScroll
        self.title = image.name
        self.image = image
        self.ax = ax
        self.index = int(len(self.image.ct_scan)/2)
        self.plane = None
        self.patch = None
        self.onSetPlane = onSetPlane
        self.ax_data = self.ax.imshow(self.image.ct_scan[self.index], cmap='gray')
        self.eventSetup()
        self.ax.set_title(f"{self.title} | {self.index}")

        self.selectedLine = (None, None)
        self.pressed = False

        def onButtonPress(event):
            if glvars.selected_axis is not self.ax:
                return

            self.pressed = True
            self.selectedLine = ((event.xdata, event.ydata), None)

        def onButtonRelease(event):
            if glvars.selected_axis is not self.ax:
                return

            self.pressed = False
            self.selectedLine = (self.selectedLine[0], (event.xdata, event.ydata))
            ((x1, y1), (x2, y2)) = self.selectedLine
            if x1 > x2:
                x1, y1, x2, y2 = x2, y2, x1, y1
            self.selectedLine = ((x1, y1), (x2, y2))
            gllines.setPoints(self.title, Point2D(x1, y1), Point2D(x2, y2))
            if self.onSetPlane:
                self.onSetPlane(self)

        def onMouseMove(event):
            if glvars.selected_axis is not self.ax:
                return

            if not self.pressed:
                return
            self.selectedLine = (self.selectedLine[0], (event.xdata, event.ydata))
            if self.patch:
                self.patch.remove()
                self.patch = None

            x, y = ((self.selectedLine[0][0], self.selectedLine[1][0]), (self.selectedLine[0][1], self.selectedLine[1][1]))
            self.patch = mlines.Line2D(x, y, lw=2, color='red', alpha=1)
            self.ax.add_line(self.patch)
            glvars.fig.canvas.draw_idle()

        glvars.fig.canvas.mpl_connect('button_press_event', onButtonPress)
        glvars.fig.canvas.mpl_connect('motion_notify_event', onMouseMove)
        glvars.fig.canvas.mpl_connect('button_release_event', onButtonRelease)
