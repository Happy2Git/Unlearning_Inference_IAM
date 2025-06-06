#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=3:59:00
#SBATCH --gpus=1
#SBATCH --constraint="v100"
#SBATCH --cpus-per-gpu=5
#SBATCH --mem=256GB
#SBATCH --job-name=all_retrain
#SBATCH --mail-type=ALL
#SBATCH --output=logs/new_records/all_retrain-%x-%j-slurm.out
#SBATCH --error=logs/new_records/all_retrain-%x-%j-slurm.err
# specificy  project dir

# source conda env
source $HOME/miniforge/bin/activate base
echo "activation conda"
conda activate IAM 
echo "activation env"

#run the application:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniforge/envs/IAM/lib/
cd $HOME/projs/Evaluate-Unlearning
python new_records.py --CLASS_init=0 --LOOP=10 --unlearn_type='one_class' --SHADOW_AVE_FLAG
python new_records.py --CLASS_init=0 --LOOP=10 --unlearn_type='class_percentage' --forget_class_ratio=0.1 --SHADOW_AVE_FLAG

