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


def enter_axes(event):
    gl.selected_axis = event.inaxes


parser = argparse.ArgumentParser()
parser.add_argument('--image', default='./Gomez_T1', type=str, help='Dataset Path')
args = parser.parse_args()


image = ItkImage(args.image)
imageNoise = ItkImage(args.image)

noise = np.random.normal(np.mean(image.ct_scan), (np.mean(image.ct_scan)/2)*0.5, imageNoise.ct_scan.shape)

imageNoise.setData(imageNoise.ct_scan + noise)

fig, (imageAX, noiseAX) = plt.subplots(1, 2)
fig.canvas.mpl_connect('axes_enter_event', enter_axes)

print(np.mean(imageNoise.ct_scan), np.mean(image.ct_scan))

plotOrig = VolumeImage(image, imageAX, fig, "image", gl)
plotNoise = VolumeImage(imageNoise, noiseAX, fig, "Noise", gl)

plt.show()
