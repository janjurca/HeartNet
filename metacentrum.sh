
# METACENTRUM
# qsub -I -q gpu -l select=1:ncpus=2:ngpus=1:mem=16gb:scratch_local=30GB:gpu_cap=cuda80 -l walltime=24:00:00
    
module add cuda-9.0
module load cudnn-7.0

cd $SCRATCHDIR || exit 1
mkdir temp
TMPDIR=$(pwd)/temp
export TMPDIR
wget https://bootstrap.pypa.io/get-pip.py
python3.7 get-pip.py
python3.7 -m pip install numpy  --cache-dir $(pwd)/cache --root $(pwd) -I
python3.7 -m pip install torch torchvision --cache-dir $(pwd)/cache --root $(pwd) -I
export PYTHONPATH=$(pwd)/usr/local/lib/python3.7/dist-packages/:$PYTHONPATH


cd /storage/brno2/home/xjurca08/storage/brno2/home/xjurca08
git clone https://github.com/janjurca/HeartNet.git
cd HeartNet
git pull

