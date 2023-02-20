from heartRotation import inference
import argparse
from utils.compute_plane import rotateImage
from utils.itkImage import ItkImage
from utils.volumeImage import VolumeImage
import matplotlib.pylab as plt
from utils.dataset import GomezT1RotationInference
from nets.vnet import VNetRegression
import torch
import numpy as np
import SimpleITK as sitk


class GL:
    selected_axis = None


gl = GL()

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='image Path')
parser.add_argument('--checkpoint', required=True, type=str, help='model file')
parser.add_argument('--checkpoint-sa', required=True, type=str, help='model file')
parser.add_argument('--checkpoint-ch4', required=True, type=str, help='model file')
parser.add_argument('--checkpoint-ch2', required=True, type=str, help='model file')
args = parser.parse_args()

inferenceSet = GomezT1RotationInference(root=args.dataset, resolution=[128, 128, 128])


def inference(dataset, checkpoint, planes=["sa", "ch4", "ch2"]):
    cuda = False
    model = VNetRegression(elu=False, nll=True, outCH=len(planes))

    print("=> loading checkpoint '{}'".format(checkpoint))
    checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])

    if cuda:
        model = model.cuda()

    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    model.eval()
    for i in range(len(dataset)):
        (data,), image = dataset.get(i)
        gtsa = image.clone()
        gtch4 = image.clone()
        gtch2 = image.clone()
        if cuda:
            data = data.cuda()
        print(data.shape)
        with torch.no_grad():
            data = torch.tensor([data.tolist()])
            output = model(data)

        output = output.view([len(planes), 128, 128, 128])
        output = output.cpu()
        output = output.detach().numpy()
        if len(planes) == 1:
            if "sa" in planes:
                sa = np.array(output[0], dtype=float)
                gtsa.setData(sa)
            if "ch4" in planes:
                ch4 = np.array(output[0], dtype=float)
                gtch4.setData(ch4)
            if "ch2" in planes:
                ch2 = np.array(output[0], dtype=float)
                gtch2.setData(ch2)
        elif len(planes) == 3:
            sa = np.array(output[0], dtype=float)
            ch4 = np.array(output[1], dtype=float)
            ch2 = np.array(output[2], dtype=float)
            gtsa.setData(sa)
            gtch4.setData(ch4)
            gtch2.setData(ch2)
        else:
            raise Exception("Planes are badly defined.")

        yield image, gtsa, gtch4, gtch2


for i, ((image, Rotsa, Rotch4, Rotch2), (_, Rotsa_solo, _, _), (_, _, Rotch4_solo, _), (_, _, _, Rotch2_solo)) in enumerate(zip(
    inference(inferenceSet, args.checkpoint),
    inference(inferenceSet, args.checkpoint_sa, ["sa"]),
    inference(inferenceSet, args.checkpoint_ch4, ["ch4"]),
    inference(inferenceSet, args.checkpoint_ch2, ["ch2"]),
)):
    print("Proccesing file:", image.filename)
    original = ItkImage(image.filename)
    Rotsa.resize(original.res())
    Rotch4.resize(original.res())
    Rotch2.resize(original.res())
    Rotsa_solo.resize(original.res())
    Rotch4_solo.resize(original.res())
    Rotch2_solo.resize(original.res())

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

    fig, ((AxSA, AxCH4,  AxCH2), (AxSA_solo, AxCH4_solo, AxCH2_solo)) = plt.subplots(2, 3)

    def enter_axes(event):
        gl.selected_axis = event.inaxes
    fig.canvas.mpl_connect('axes_enter_event', enter_axes)

    # plotOrig = VolumeImage(original, Axoriginal, fig, "Original", gl)

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

    fig, (axx) = plt.subplots(1, 1)
    plotSA = VolumeImage(imageSA, axx, fig, "SA MultiPlane", gl)
    plotSA.setIndex(indexSA)
    plt.savefig("sa_MultiPlane.jpg")

    fig, (axx) = plt.subplots(1, 1)
    plotCH4 = VolumeImage(imageCH4, axx, fig, "CH4 MultiPlane", gl)
    plotCH4.setIndex(indexCH4)
    plt.savefig("ch4_MultiPlane.jpg")

    fig, (axx) = plt.subplots(1, 1)
    plotCH2 = VolumeImage(imageCH2, axx, fig, "CH2 MultiPlane", gl)
    plotCH2.setIndex(indexCH2)
    plt.savefig("ch2_MultiPlane.jpg")

    fig, (axx) = plt.subplots(1, 1)
    plotSA_solo = VolumeImage(imageSA_solo, axx, fig, "SA SinglePlane", gl)
    plotSA_solo.setIndex(indexSA_solo)
    plt.savefig("sa_SinglePlane.jpg")

    fig, (axx) = plt.subplots(1, 1)
    plotCH4_solo = VolumeImage(imageCH4_solo, axx, fig, "CH4 SinglePlane", gl)
    plotCH4_solo.setIndex(indexCH4_solo)
    plt.savefig("ch4_SinglePlane.jpg")

    fig, (axx) = plt.subplots(1, 1)
    plotCH2_solo = VolumeImage(imageCH2_solo, axx, fig, "CH2 SinglePlane", gl)
    plotCH2_solo.setIndex(indexCH2_solo)
    plt.savefig("ch2_SinglePlane.jpg")

    plt.show()
