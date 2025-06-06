import os

# List of datasets, unlearning methods, and class ratios
datasets = ['cinic10']

# SLURM template
slurm_template = """#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=11:59:00
#SBATCH --gpus=1
#SBATCH --constraint="a100|v100"
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=128GB
#SBATCH --job-name={job_name}
#SBATCH --mail-type=ALL
#SBATCH --output=logs/shadows/{output_name}-%x-%j-slurm.out
#SBATCH --error=logs/shadows/{output_name}-%x-%j-slurm.err

# source conda env
source $HOME/miniforge/bin/activate base
echo "activation conda"
conda activate EvaUnle 
echo "activation env"

#run the application:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/mambaforge/envs/EvaUnle/lib/
cd $HOME/projs/Evaluate-Unlearning

{python_commands}
"""

# Directory to save SLURM scripts
slurm_dir = "slurm_scripts/shadow"
os.makedirs(slurm_dir, exist_ok=True)

def generate_python_command(dataname, shadow_index = None):
    # make the shadow_index like '0 1 2 3 4 5 6 7', not '[0,1,2,3,4,5,6,7]' in the command
    shadow_index = ' '.join(map(str, shadow_index))
    return f"python shadow_training.py --dataname '{dataname}' --shadow_idx {shadow_index} --origin True --VERBOSE True"

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
    # os.system(f"sbatch {script_name}")

# Main process to generate and submit jobs
batch_size = 10  # Number of python commands per SLURM task
current_batch = []
batch_counter = 0
dataname = 'cinic10'
# 2. One Class unlearning
model_nums = 128
# creata a batch for shadow_index from 0 to 127, in 16 batches
BATCH = 16
for batch in range(BATCH):
    shadow_index = list(range(batch*8, (batch+1)*8))
    command = generate_python_command(dataname, shadow_index=shadow_index)
    job_name = f'shadow_{batch_counter}_remaining'
    output_name = f'shadow_training_{batch}_{job_name}'
    create_and_submit_slurm_job(job_name, output_name, [command])
    batch_counter +=1
    current_batch.append(command)

