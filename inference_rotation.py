from heartRotation import inference
import argparse
import SimpleITK as sitk
from utils.compute_plane import rotateImage
from utils.itkImage import ItkImage
from utils.volumeImage import VolumeImage
import matplotlib.pylab as plt


class GL:
    selected_axis = None


gl = GL()

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='./Gomez_T1', type=str, help='Dataset Path')
parser.add_argument('--checkpoint', default='./checkpoint.pth.tar', type=str, help='model file')
args = parser.parse_args()

for image, gtsa, gtch4, gtch2 in inference.inference(args.dataset, args.checkpoint):
    sitk.Show(gtsa.image)
    original = ItkImage(image.filename)
    gtsa.resize(original.res())
    rotated, index = rotateImage(original.clone(), gtsa)
    fig, (Axoriginal, AxSA) = plt.subplots(1, 2)

    def enter_axes(event):
        gl.selected_axis = event.inaxes
    fig.canvas.mpl_connect('axes_enter_event', enter_axes)

    plotOrig = VolumeImage(original, Axoriginal, fig, "Original", gl)
    plotSA = VolumeImage(rotated, AxSA, fig, "SA", gl)
    plotSA.setIndex(index)
    plt.show()
