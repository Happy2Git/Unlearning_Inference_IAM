#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=3:59:00
#SBATCH --gpus=1
#SBATCH --constraint="a100|v100"
#SBATCH --cpus-per-gpu=5
#SBATCH --mem=256GB
#SBATCH --job-name=record_refnums
#SBATCH --mail-type=ALL
#SBATCH --output=logs/new_records/all_ref_numbs-%x-%j-slurm.out
#SBATCH --error=logs/new_records/all_ref_numbs-%x-%j-slurm.err
# specificy  project dir

# source conda env
source $HOME/miniforge/bin/activate base
echo "activation conda"
conda activate IAM 
echo "activation env"

#run the application:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniforge/envs/IAM/lib/
cd $HOME/projs/Evaluate-Unlearning

python new_records.py --SEED_init=42 --LOOP=10 --unlearn_type='set_random' --model_numbs=128 --SHADOW_AVE_FLAG

python new_records.py --SEED_init=42 --LOOP=10 --unlearn_type='set_random' --model_numbs=1 --SHADOW_AVE_FLAG

python new_records.py --SEED_init=42 --LOOP=10 --unlearn_type='set_random' --model_numbs=2 --SHADOW_AVE_FLAG

python new_records.py --SEED_init=42 --LOOP=10 --unlearn_type='set_random' --model_numbs=4 --SHADOW_AVE_FLAG

python new_records.py --SEED_init=42 --LOOP=10 --unlearn_type='set_random' --model_numbs=8 --SHADOW_AVE_FLAG

python new_records.py --SEED_init=42 --LOOP=10 --unlearn_type='set_random' --model_numbs=16 --SHADOW_AVE_FLAG

python new_records.py --SEED_init=42 --LOOP=10 --unlearn_type='set_random' --model_numbs=32 --SHADOW_AVE_FLAG

python new_records.py --SEED_init=42 --LOOP=10 --unlearn_type='set_random' --model_numbs=64 --SHADOW_AVE_FLAG



# python new_records.py --CLASS_init=0 --LOOP=10 --unlearn_type='one_class'
# python new_records.py --CLASS_init=0 --LOOP=10 --unlearn_type='class_percentage' --forget_class_ratio=0.3
# python new_records.py --CLASS_init=0 --LOOP=10 --unlearn_type='class_percentage' --forget_class_ratio=0.5
# python new_records.py --CLASS_init=0 --LOOP=10 --unlearn_type='class_percentage' --forget_class_ratio=0.7
# python new_records.py --CLASS_init=0 --LOOP=10 --unlearn_type='class_percentage' --forget_class_ratio=0.9
# python new_records_internal.py --BREAKs=20
#     base_args.update({'forget_class': args_input.forget_class, 'unlearn_type': args_input.unlearn_type, 'forget_class_ratio': args_input.forget_class_ratio})


