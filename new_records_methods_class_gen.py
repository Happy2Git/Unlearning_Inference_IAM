import os

# List of datasets, unlearning methods, and class ratios
datasets = ['cifar10', 'cinic10',  'cifar100', 'purchase']

# SLURM template
slurm_template = """#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=3:59:00
#SBATCH --gpus=1
#SBATCH --constraint="a100|v100"
#SBATCH --cpus-per-gpu=5
#SBATCH --mem=256GB
#SBATCH --job-name=class
#SBATCH --mail-type=ALL
#SBATCH --output=logs/new_records/methods/class-%x-%j-slurm.out
#SBATCH --error=logs/new_records/methods/class-%x-%j-slurm.err
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
slurm_dir = "records_scripts/new_records"
os.makedirs(slurm_dir, exist_ok=True)

# --unlearn_type='class_percentage' --forget_class_ratio=0.1
def generate_python_command(dataname):
    commands = f"""python new_records_method_class.py --CLASS_init=0 --LOOP=10 --unlearn_type='one_class' --dataname='{dataname}' --unlearn_method='all'  --metrics 'p_interapprox'  --SHADOW_AVE_FLAG
"""
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
# 2. One Class unlearning
model_nums = 128
# creata a batch for shadow_index from 0 to 127, in 16 batches
BATCH = 16
for batch, dataname in enumerate(datasets):
    command = generate_python_command(dataname)
    job_name = f'class_{batch_counter}_remaining'
    output_name = f'class_training_{batch}_{job_name}'
    create_and_submit_slurm_job(job_name, output_name, [command])
    batch_counter +=1
    current_batch.append(command)

