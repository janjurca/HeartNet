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
    # sitk.Show(gtsa.image)
    # sitk.Show(gtch4.image)
    # sitk.Show(gtch2.image)

    original = ItkImage(image.filename)
    gtsa.resize(original.res())
    gtch4.resize(original.res())
    gtch2.resize(original.res())
    imageCH4, indexCH4 = rotateImage(ItkImage(original.filename), gtch4)
    imageCH2, indexCH2 = rotateImage(ItkImage(original.filename), gtch2)
    imageSA, indexSA = rotateImage(ItkImage(original.filename), gtsa)
    fig, (Axoriginal, AxSA, AxCH4, AxCH2) = plt.subplots(1, 4)

    sitk.Show(ItkImage(original.filename).image)
    sitk.Show(imageCH4.image)

    def enter_axes(event):
        gl.selected_axis = event.inaxes
    fig.canvas.mpl_connect('axes_enter_event', enter_axes)

    plotOrig = VolumeImage(original, Axoriginal, fig, "Original", gl)
    plotSA = VolumeImage(imageSA, AxSA, fig, "SA", gl)
    plotSA.setIndex(indexSA)
    plotCH4 = VolumeImage(imageCH4, AxCH4, fig, "CH4", gl)
    plotCH4.setIndex(indexCH4)
    plotCH2 = VolumeImage(imageCH2, AxCH2, fig, "CH2", gl)
    plotCH2.setIndex(indexCH2)
    plt.show()
