import os

# List of datasets, unlearning arch, and class ratios
model_archs = ['ResNet18', 'ResNet34', 'ResNet50', 'vgg16', 'vgg11', 'DenseNet']
# SLURM template
slurm_template = """#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=0:59:00
#SBATCH --gpus=1
#SBATCH --constraint="a100|v100"
#SBATCH --cpus-per-gpu=5
#SBATCH --mem=256GB
#SBATCH --job-name=archq
#SBATCH --mail-type=ALL
#SBATCH --output=logs/new_records/arch/query-%x-%j-slurm.out
#SBATCH --error=logs/new_records/arch/query-%x-%j-slurm.err
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
slurm_dir = "records_scripts/arch"
os.makedirs(slurm_dir, exist_ok=True)

def generate_python_command(model_arch):
    commands = f"""python new_records_shift_model.py --SEED_init=42 --LOOP=1 --unlearn_type='set_random' --model_arch='{model_arch}' --SHADOW_AVE_FLAG"""
    return commands


# Function to create and submit SLURM scripts with 10 tasks per SLURM file
def create_and_submit_slurm_job(job_name, output_name, python_commands, model_arch):
    slurm_script_content = slurm_template.format(
        job_name=job_name,
        output_name=output_name,
        python_commands="\n".join(python_commands)
    )
    
    if model_arch == 'shufflenet' or model_arch == 'swin_t':
        slurm_script_content = slurm_script_content.replace("#SBATCH --constraint=\"a100|v100\"", "#SBATCH --constraint=\"a100\"")

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
for batch, model_arch in enumerate(model_archs):
    command = generate_python_command(model_arch)
    job_name = f'arch_{batch_counter}_remaining'
    output_name = f'arch_training_{batch}_{job_name}'
    create_and_submit_slurm_job(job_name, output_name, [command], model_arch)
    batch_counter +=1
    current_batch.append(command)

