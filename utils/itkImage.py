import SimpleITK as sitk
import numpy as np


class ItkImage:
    def __init__(self, filename: str, resolution=None) -> None:
        self.filename = filename
        self.resolution = resolution
        self.augment_mhd_file()
        self.load()
        self.heartBox = {}

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
        if self.resolution:
            original_CT = self.image
            dimension = original_CT.GetDimension()
            reference_physical_size = np.zeros(original_CT.GetDimension())
            reference_physical_size[:] = [(sz-1)*spc if sz*spc > mx else mx for sz, spc, mx in zip(original_CT.GetSize(), original_CT.GetSpacing(), reference_physical_size)]

            reference_origin = original_CT.GetOrigin()
            reference_direction = original_CT.GetDirection()

            reference_size = self.resolution
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
        self.heartBox = None
        self.refresh()

    def refresh(self) -> None:
        # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
        self.ct_scan = sitk.GetArrayFromImage(self.image)
        # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
        self.origin = np.array(list(reversed(self.image.GetOrigin())))  # TODO handle different rotations
        # Read the spacing along each dimension
        self.spacing = np.array(list(reversed(self.image.GetSpacing())))

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
        return sitk.Resample(self.image, reference_image, transform,
                             interpolator, default_value)

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
        self.load()
        theta_x = np.deg2rad(theta_x)
        theta_y = np.deg2rad(theta_y)
        theta_z = np.deg2rad(theta_z)
        euler_transform = sitk.Euler3DTransform(self.get_center(), theta_x, theta_y, theta_z, (0, 0, 0))
        self.image = self.resample(euler_transform)
        self.refresh()

    def translate(self, x, y, z):
        h = self.heartBox
        self.load()
        self.heartBox = h
        transform = sitk.TranslationTransform(3, (x, y, z))
        self.image = self.resample(transform)
        self.refresh()
        if self.heartBox:
            self.heartBox["left"] = self.heartBox["left"] + z
            self.heartBox["right"] = self.heartBox["right"] + z
            self.heartBox["top"] = self.heartBox["top"] + y
            self.heartBox["bottom"] = self.heartBox["bottom"] + y
            self.heartBox["front"] = self.heartBox["front"] + x
            self.heartBox["back"] = self.heartBox["back"] + x

    def setHeartBox(self, left, right, top, bottom, front, back):
        self.heartBox = {}
        self.heartBox["left"] = left
        self.heartBox["right"] = right
        self.heartBox["top"] = top
        self.heartBox["bottom"] = bottom
        self.heartBox["front"] = front
        self.heartBox["back"] = back
