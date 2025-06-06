import os

# List of datasets, unlearning refnum, and class ratios
datasets = ['cifar10', 'cinic10',  'cifar100', 'purchase']
model_nums = [128, 1, 2, 4, 8, 16, 32, 64]

# SLURM template
slurm_template = """#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=3:59:00
#SBATCH --gpus=1
#SBATCH --constraint="a100|v100"
#SBATCH --cpus-per-gpu=5
#SBATCH --mem=256GB
#SBATCH --job-name=refnumq
#SBATCH --mail-type=ALL
#SBATCH --output=logs/new_records/refnum/ref-%x-%j-slurm.out
#SBATCH --error=logs/new_records/refnum/ref-%x-%j-slurm.err
# specificy  project dir

# source conda env
source $HOME/miniforge/bin/activate base
echo "activation conda"
conda activate IAM 
echo "activation env"

#run the application:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniforge/envs/IAM/lib/
cd $HOME/projs/Evaluate-Unlearning

{python_commands}
"""

# Directory to save SLURM scripts
slurm_dir = "records_scripts/ref_nums"
os.makedirs(slurm_dir, exist_ok=True)

def generate_python_command(dataset, model_num):
    commands = f"""python new_records.py --SEED_init=42 --LOOP=10 --dataname='{dataset}' --unlearn_type='set_random' --model_numbs={model_num}  --SHADOW_AVE_FLAG """
    
    return commands


# Function to create and submit SLURM scripts with 10 tasks per SLURM file
def create_and_submit_slurm_job(job_name, output_name, python_commands):
    slurm_script_content = slurm_template.format(
        job_name=job_name,
        output_name=output_name,
        python_commands="\n".join(python_commands)
    )
    
    # Save the SLURM script
    script_name = f"{slurm_dir}/slurm_{job_name}.sh"
    with open(script_name, 'w') as f:
        f.write(slurm_script_content)
    
    # Submit the SLURM script
    os.system(f"sbatch {script_name}")

# Main process to generate and submit jobs
current_batch = []
batch_counter = 0
# creata a batch for shadow_index from 0 to 127, in 16 batches
BATCH = 16
for dataset in datasets:
    for model_num in model_nums:
        command = generate_python_command(dataset, model_num)
        job_name = f'ref_{batch_counter}_remaining'
        output_name = f'ref_num_{job_name}'
        create_and_submit_slurm_job(job_name, output_name, [command])
        batch_counter +=1
        current_batch.append(command)

