#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=23:59:00
#SBATCH --gpus=1
#SBATCH --constraint="a100|v100"
#SBATCH --cpus-per-gpu=5
#SBATCH --mem=256GB
#SBATCH --job-name=single
#SBATCH --mail-type=ALL
#SBATCH --output=logs/shadow/single-%x-%j-slurm.out
#SBATCH --error=logs/shadow/single-%x-%j-slurm.err
# specificy  project dir


# source conda env
source $HOME/miniforge/bin/activate base
echo "activation conda"
conda activate UnInf_IAM 
echo "activation env"

#run the application:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniforge/envs/UnInf_IAM/lib/
cd $HOME/projs/Evaluate-Unlearning
python shadow_training.py --dataname 'cifar100' --ratio 0.8 --shadow_idx 0 --origin True --VERBOSE True
# choices=['ResNet18', 'ResNet34', 'ResNet50', 'vgg16', 'shufflenet', 'DenseNet', 'MobileNetV2', 'EfficientNet', 'swin_t'],) 
# python shadow_training.py --dataname 'cifar100' --ratio 0.8 --shadow_idx 0 --origin True --VERBOSE True --ARCH=True --arch_name='swin_t'
# python shadow_training.py --dataname 'cifar100' --ratio 0.8 --shadow_idx 0 --origin True --VERBOSE True --ARCH=True --arch_name='ResNet18'
# python shadow_training.py --dataname 'cifar100' --ratio 0.8 --shadow_idx 0 --origin True --VERBOSE True --ARCH=True --arch_name='ResNet34'
# python shadow_training.py --dataname 'cifar100' --ratio 0.8 --shadow_idx 0 --origin True --VERBOSE True --ARCH=True --arch_name='ResNet50'
# python shadow_training.py --dataname 'cifar100' --ratio 0.8 --shadow_idx 0 --origin True --VERBOSE True --ARCH=True --arch_name='vgg16'
# python shadow_training.py --dataname 'cifar100' --ratio 0.8 --shadow_idx 0 --origin True --VERBOSE True --ARCH=True --arch_name='shufflenet'
# python shadow_training.py --dataname 'cifar100' --ratio 0.8 --shadow_idx 0 --origin True --VERBOSE True --ARCH=True --arch_name='DenseNet'
# python shadow_training.py --dataname 'cifar100' --ratio 0.8 --shadow_idx 0 --origin True --VERBOSE True --ARCH=True --arch_name='MobileNetV2'
# python shadow_training.py --dataname 'cifar100' --ratio 0.8 --shadow_idx 0 --origin True --VERBOSE True --ARCH=True --arch_name='EfficientNet'


