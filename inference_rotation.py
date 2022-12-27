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

inferenceSet = GomezT1Rotation(root=args.dataset, portion=0.2, resolution=[128, 128, 128])

for i, ((image, Rotsa, Rotch4, Rotch2), (_, Rotsa_solo, _, _), (_, _, Rotch4_solo, _), (_, _, _, Rotch2_solo)) in enumerate(zip(
    inference.inference(inferenceSet, args.checkpoint),
    inference.inference(inferenceSet, args.checkpoint_sa, ["sa"]),
    inference.inference(inferenceSet, args.checkpoint_ch4, ["ch4"]),
    inference.inference(inferenceSet, args.checkpoint_ch2, ["ch2"]),
)):
    (_, _), _, gtsa, gtch4, gtch2 = inferenceSet.get(i)

    gtsa = gtsa.clone()
    gtch4 = gtch4.clone()
    gtch2 = gtch2.clone()
    print("Proccesing file:", image.filename)
    original = ItkImage(image.filename)
    Rotsa.resize(original.res())
    Rotch4.resize(original.res())
    Rotch2.resize(original.res())
    Rotsa_solo.resize(original.res())
    Rotch4_solo.resize(original.res())
    Rotch2_solo.resize(original.res())

    gtsa.resize(original.res())
    gtch4.resize(original.res())
    gtch2.resize(original.res())

    print("Proccesing multishot results")
    print("CH4")
    imageCH4, indexCH4 = rotateImage(ItkImage(original.filename), Rotch4)
    print("CH2")
    imageCH2, indexCH2 = rotateImage(ItkImage(original.filename), Rotch2)
    print("SA")
    imageSA, indexSA = rotateImage(ItkImage(original.filename), Rotsa)

    print("Proccesing solo results")
    print("CH4")
    imageCH4_solo, indexCH4_solo = rotateImage(ItkImage(original.filename), Rotch4_solo)
    print("CH2")
    imageCH2_solo, indexCH2_solo = rotateImage(ItkImage(original.filename), Rotch2_solo)
    print("SA")
    imageSA_solo, indexSA_solo = rotateImage(ItkImage(original.filename), Rotsa_solo)

    print("Proccesing GTS")
    print("CH4")
    imageCH4_gt, indexCH4_gt = rotateImage(ItkImage(original.filename), gtch4)
    print("CH2")
    imageCH2_gt, indexCH2_gt = rotateImage(ItkImage(original.filename), gtch2)
    print("SA")
    imageSA_gt, indexSA_gt = rotateImage(ItkImage(original.filename), gtsa)

    fig, ((AxSA_gt, AxCH4_gt, AxCH2_gt), (AxSA, AxCH4,  AxCH2), (AxSA_solo, AxCH4_solo, AxCH2_solo)) = plt.subplots(3, 3)

    def enter_axes(event):
        gl.selected_axis = event.inaxes
    fig.canvas.mpl_connect('axes_enter_event', enter_axes)

    #plotOrig = VolumeImage(original, Axoriginal, fig, "Original", gl)

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

    plotSA_gt = VolumeImage(imageSA_gt, AxSA_gt, fig, "SA_gt", gl)
    plotSA_gt.setIndex(indexSA_gt)
    plotCH4_gt = VolumeImage(imageCH4_gt, AxCH4_gt, fig, "CH4_gt", gl)
    plotCH4_gt.setIndex(indexCH4_gt)
    plotCH2_gt = VolumeImage(imageCH2_gt, AxCH2_gt, fig, "CH2_gt", gl)
    plotCH2_gt.setIndex(indexCH2_solo)

    plt.show()
