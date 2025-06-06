#!/bin/bash
#SBATCH --account=conf-neurips-2025.05.22-wangd0d
#SBATCH --nodes=1
#SBATCH --time=3:59:00
#SBATCH --gpus=3
#SBATCH --constraint="a100"
#SBATCH --cpus-per-gpu=5
#SBATCH --mem=256GB
#SBATCH --job-name=negative_label_traps_ip
#SBATCH --mail-type=ALL
#SBATCH --output=logs/%x-%j-slurm.out
#SBATCH --error=logs/%x-%j-slurm.err
# specificy  project dir


# source conda env
module load cuda/12.1
source $HOME/miniforge/bin/activate base
echo "activation conda"
conda activate mia_llm 
echo "activation env"

#run the application:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/mambaforge/envs/mia_llm/lib/
cd $HOME/projs/Evaluate-Unlearning
/mia_llms_benchmark
python main_data_parallel_adv.py -c exp_configs/config_template_traps_ip_adv.yaml --run-all \
    --slurm_name "$SLURM_JOB_NAME" --slurm_id "$SLURM_JOB_ID" \
    --output="$HOME/projs/Evaluate-Unlearning/mia_llms_benchmark/exp_output/output_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.pkl" \
 