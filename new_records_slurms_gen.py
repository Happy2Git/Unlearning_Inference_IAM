import os

datasets = ['cifar10', 'cinic10',  'cifar100', 'purchase']
unlearn_types = ['class_percentage']
class_ratio = [0.1]
# SLURM template
slurm_template = """#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=0:59:00
#SBATCH --gpus=1
#SBATCH --constraint="a100|v100"
#SBATCH --cpus-per-gpu=5
#SBATCH --mem=256GB
#SBATCH --job-name={job_name}
#SBATCH --mail-type=ALL
#SBATCH --output=logs/new_records/slurm/retrain-{output_name}-%x-%j-slurm.out
#SBATCH --error=logs/new_records/slurm/retrain-{output_name}-%x-%j-slurm.err
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
slurm_dir = "records_scripts"
os.makedirs(slurm_dir, exist_ok=True)

def generate_python_command(unlearn_types, count, dataname):
    if unlearn_types == 'set_random':
        return f"python new_records.py --SEED_init={count} --LOOP=1 --unlearn_type=set_random"
    elif unlearn_types == 'one_class':
        return f"python new_records.py --CLASS_init={count} --LOOP=1 --unlearn_type=one_class"
    elif unlearn_types == 'class_percentage':
        return f"python new_records.py --CLASS_init={count} --forget_class_ratio=0.3 --dataname='{dataname}' --LOOP=1 --unlearn_type=class_percentage"

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
for class_index in range(10):
    for dataname in datasets:
        command = generate_python_command('class_percentage', class_index, dataname)
        job_name = f'batch_{batch_counter}_remaining'
        output_name = f'class_percentage_{class_index}_{dataname}_{job_name}'
        create_and_submit_slurm_job(job_name, output_name, [command])
        batch_counter +=1
        current_batch.append(command)

