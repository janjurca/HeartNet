from heartRotation import inference
import argparse
import SimpleITK as sitk
from utils.compute_plane import rotateImage
from utils.itkImage import ItkImage
from utils.volumeImage import VolumeImage
import matplotlib.pylab as plt
from utils.dataset import GomezT1Rotation


class GL:
    selected_axis = None


gl = GL()

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='./Gomez_T1', type=str, help='Dataset Path')
parser.add_argument('--checkpoint', required=True, type=str, help='model file')
parser.add_argument('--checkpoint-sa', required=True, type=str, help='model file')
parser.add_argument('--checkpoint-ch4', required=True, type=str, help='model file')
parser.add_argument('--checkpoint-ch2', required=True, type=str, help='model file')
args = parser.parse_args()

inferenceSet = GomezT1Rotation(root=args.dataset, portion=0.05, resolution=[128, 128, 128])

for (image, gtsa, gtch4, gtch2), (_, gtsa_solo, _, _), (_, _, gtch4_solo, _), (_, _, _, gtch2_solo) in zip(
    inference.inference(inferenceSet, args.checkpoint),
    inference.inference(inferenceSet, args.checkpoint_sa, ["sa"]),
    inference.inference(inferenceSet, args.checkpoint_ch4, ["ch4"]),
    inference.inference(inferenceSet, args.checkpoint_ch2, ["ch2"]),
):

    original = ItkImage(image.filename)
    gtsa.resize(original.res())
    gtch4.resize(original.res())
    gtch2.resize(original.res())
    gtsa_solo.resize(original.res())
    gtch4_solo.resize(original.res())
    gtch2_solo.resize(original.res())

    imageCH4, indexCH4 = rotateImage(ItkImage(original.filename), gtch4)
    imageCH2, indexCH2 = rotateImage(ItkImage(original.filename), gtch2)
    imageSA, indexSA = rotateImage(ItkImage(original.filename), gtsa)

    imageCH4_solo, indexCH4_solo = rotateImage(ItkImage(original.filename), gtch4_solo)
    imageCH2_solo, indexCH2_solo = rotateImage(ItkImage(original.filename), gtch2_solo)
    imageSA_solo, indexSA_solo = rotateImage(ItkImage(original.filename), gtsa_solo)

    fig, (Axoriginal, AxSA, AxCH4, AxCH2, AxOriginal_solo, AxSA_solo, AxCH4_solo, AxCH2_solo) = plt.subplots(1, 8)

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

    plotSA_solo = VolumeImage(imageSA_solo, AxSA_solo, fig, "SA_solo", gl)
    plotSA_solo.setIndex(indexSA_solo)
    plotCH4_solo = VolumeImage(imageCH4_solo, AxCH4_solo, fig, "CH4_solo", gl)
    plotCH4_solo.setIndex(indexCH4_solo)
    plotCH2_solo = VolumeImage(imageCH2_solo, AxCH2_solo, fig, "CH2_solo", gl)
    plotCH2_solo.setIndex(indexCH2_solo)

    plt.show()
