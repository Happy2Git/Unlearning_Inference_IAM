#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=11:59:00
#SBATCH --gpus=1
#SBATCH --constraint="a100|v100"
#SBATCH --cpus-per-gpu=5
#SBATCH --mem=256GB
#SBATCH --job-name=shift_inter
#SBATCH --mail-type=ALL
#SBATCH --output=logs/new_records/shift_appro-%x-%j-slurm.out
#SBATCH --error=logs/new_records/shift_appro-%x-%j-slurm.err
# specificy  project dir

# source conda env
source $HOME/miniforge/bin/activate base
echo "activation conda"
conda activate IAM 
echo "activation env"

#run the application:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniforge/envs/IAM/lib/
cd $HOME/projs/Evaluate-Unlearning
# python new_records_internal_shift.py --BREAKs=20 --dataname='cifar10' --init_idx=0 --LOOP=10
python new_records_internal_shift.py --BREAKs=20 --dataname='cinic10' --init_idx=0 --LOOP=10  --SHADOW_AVE_FLAG
python new_records_internal_shift.py --BREAKs=20 --dataname='cifar10' --init_idx=0 --LOOP=10  --SHADOW_AVE_FLAG
#     base_args.update({'forget_class': args_input.forget_class, 'unlearn_type': args_input.unlearn_type, 'forget_class_ratio': args_input.forget_class_ratio})


