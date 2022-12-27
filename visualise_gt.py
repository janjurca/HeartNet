from heartRotation import inference
import argparse
import SimpleITK as sitk
from utils.compute_plane import rotateImage
from utils.itkImage import ItkImage
from utils.volumeImage import VolumeImage
import matplotlib.pylab as plt
from utils.dataset import GomezT1Rotation
import numpy as np
import torch
from torch.nn.functional import normalize


class GL:
    selected_axis = None


gl = GL()

parser = argparse.ArgumentParser()
parser.add_argument('--image', default='./Gomez_T1', type=str, help='Dataset Path')
parser.add_argument('--gt', default='./Gomez_T1', type=str, help='Dataset Path')

args = parser.parse_args()

image = ItkImage(args.image)
gt = ItkImage(args.gt)


imageRotated, index = rotateImage(image, gt)


fig, (imageAX, gtAX) = plt.subplots(1, 2)


def enter_axes(event):
    gl.selected_axis = event.inaxes


fig.canvas.mpl_connect('axes_enter_event', enter_axes)

# plotOrig = VolumeImage(original, Axoriginal, fig, "Original", gl)

plotSA = VolumeImage(imageRotated, imageAX, fig, "image", gl)
plotSA.setIndex(index)
plotCH4 = VolumeImage(gt, gtAX, fig, "gt", gl)

print(np.max(imageRotated.ct_scan))

plt.show()
