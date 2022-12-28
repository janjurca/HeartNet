from heartRotation import inference
import argparse
import SimpleITK as sitk
from utils.compute_plane import rotateImage, comparePlanes
from utils.itkImage import ItkImage
from utils.volumeImage import VolumeImage
import matplotlib.pylab as plt
from utils.dataset import GomezT1Rotation
import numpy as np


def handleAngle(angle):
    if angle > 90:
        return 180-angle
    else:
        return angle


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

inferenceSet = GomezT1Rotation(root=args.dataset, portion=1, resolution=[128, 128, 128])

angle_results = []

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

    print("Proccesing multishot results", image.filename)
    print("Plane\t Multishot\t Solo",)
    ch4_solo_angle, ch2_solo_angle, sa_solo_angle = comparePlanes(Rotch4_solo, gtch4),  comparePlanes(Rotch2_solo, gtch2), comparePlanes(Rotsa_solo, gtsa)
    ch4_multi_angle, ch2_multi_angle, sa_multi_angle = comparePlanes(Rotch4, gtch4),  comparePlanes(Rotch2, gtch2), comparePlanes(Rotsa, gtsa)

    ch4_solo_angle = handleAngle(ch4_solo_angle)
    ch2_solo_angle = handleAngle(ch2_solo_angle)
    sa_solo_angle = handleAngle(sa_solo_angle)
    ch4_multi_angle = handleAngle(ch4_multi_angle)
    ch2_multi_angle = handleAngle(ch2_multi_angle)
    sa_multi_angle = handleAngle(sa_multi_angle)
    print(f"CH4 \t {ch4_multi_angle:.4f} \t  {ch4_solo_angle:.4f}")
    print(f"CH2 \t {ch2_multi_angle:.4f} \t  {ch2_solo_angle:.4f}")
    print(f"SA \t {sa_multi_angle:.4f} \t  {sa_solo_angle:.4f}")
    angle_results.append([ch4_solo_angle, ch2_solo_angle, sa_solo_angle, ch4_multi_angle, ch2_multi_angle, sa_multi_angle])


print(angle_results)
print(np.mean(angle_results, axis=0))
