
# METACENTRUM
# qsub -I -q gpu -l select=1:ncpus=2:ngpus=1:mem=16gb:scratch_local=30GB:gpu_cap=cuda80 -l walltime=24:00:00

module add cuda-9.0
module load cudnn-7.0
module add python36-modules-gcc

cd $SCRATCHDIR || exit 1
mkdir temp
TMPDIR=$(pwd)/temp
export TMPDIR
pip3.6 install torch torchvision SimpleITK simpy scipy --cache-dir $(pwd)/cache --root $(pwd) -I
export PYTHONPATH=$(pwd)/software/python-3.6.2/gcc/lib/python3.6/site-packages/:$PYTHONPATH

cd /storage/brno2/home/xjurca08/storage/brno2/home/xjurca08/HeartNet
git pull

python3.6 train.py --dataset /storage/brno2/home/xjurca08/storage/brno2/home/xjurca08/Gomez_T1/ --batchSz 4
