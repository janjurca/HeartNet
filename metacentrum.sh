
# METACENTRUM
# qsub -I -q gpu -l select=1:ncpus=2:ngpus=1:mem=16gb:scratch_local=30GB:gpu_cap=cuda80 -l walltime=24:00:00


#module add cuda-9.0
#module load cudnn-7.0

module load cudnn-7.6.4-cuda10.1

cd $SCRATCHDIR || exit 1
mkdir temp
TMPDIR=$(pwd)/temp
export TMPDIR
wget https://bootstrap.pypa.io/get-pip.py
python3.9 get-pip.py
python3.9 -m pip install sklearn  matplotlib  numpy regex wget 
python3.9 -m pip install torch torchvision SimpleITK sympy scipy --cache-dir $(pwd)/cache --root $(pwd) -I
export PYTHONPATH=$(pwd)/usr/local/lib/python3.9/dist-packages/:$PYTHONPATH

cd /storage/brno2/home/xjurca08/storage/brno2/home/xjurca08/HeartNet
git pull

cp -r /storage/brno2/home/xjurca08/storage/brno2/home/xjurca08/HeartNet/Gomez_T1 $SCRATCHDIR/


#python3.9 train.py --dataset /storage/brno2/home/xjurca08/storage/brno2/home/xjurca08/Gomez_T1/ --batchSz 4

python3.9 detection.py --dataset /storage/brno2/home/xjurca08/storage/brno2/home/xjurca08/HeartNet/Gomez_T1/ --batchSz 10


python3.9 rotation.py --dataset $SCRATCHDIR/Gomez_T1/ --batchSz 10 --planes sa --augment 30
python3.9 rotation.py --dataset $SCRATCHDIR/Gomez_T1/ --batchSz 10 --planes ch4 --augment 30
python3.9 rotation.py --dataset $SCRATCHDIR/Gomez_T1/ --batchSz 10 --planes ch2 --augment 30
python3.9 rotation.py --dataset $SCRATCHDIR/Gomez_T1/ --batchSz 10 --planes sa,ch4,ch2 --augment 30
