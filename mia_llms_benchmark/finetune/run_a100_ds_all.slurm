#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=11:59:00
#SBATCH --gpus=4
#SBATCH --constraint="a100"
#SBATCH --cpus-per-gpu=5
#SBATCH --mem=256GB
#SBATCH --job-name=ft_all
#SBATCH --mail-type=ALL
#SBATCH --output=logs/%x-%j-slurm.out
#SBATCH --error=logs/%x-%j-slurm.err
# specificy  project dir


# source conda env
source $HOME/miniforge/bin/activate base
echo "activation conda"
module load cuda/12.1
conda activate mia_llm 
echo "activation env"
#run the application:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/mambaforge/envs/mia_llm/lib/

echo "CUDA_HOME: $CUDA_HOME"
if [ -z "$CUDA_HOME" ]; then
    echo "CUDA_HOME is not set. Setting it to default path..."
    export CUDA_HOME=/sw/rl9g/cuda/12.1/rl9_binary
    echo "CUDA_HOME set to $CUDA_HOME"
else
    echo "CUDA_HOME is already set to $CUDA_HOME"
fi

cd $HOME/projs/Evaluate-Unlearning
/mia_llms_benchmark/finetune
# python fine_tune_ckpt_original.py
python fine_tune_ckpt_original.py --output_dir='/ibex/project/c2283/Llama-2-7b-ft-muse-news/llama2-7b-muse-news-original_new'
