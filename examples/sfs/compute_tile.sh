#!/bin/bash

# The first argument to the script
reg=${1}
tid=${2}
sel=${3}
steps=${4}
nnodes=${5}

# Create a unique job name using the argument
job_name="sfs_${reg}_${tid}"

# Generate the SLURM script with the desired job name
mkdir .tmp_slurm/
cat << EOF > .tmp_slurm/_slurm_script_${reg}_${tid}_${sel}_${steps}.sh
#!/usr/bin/env bash
#SBATCH --account=j1010
#SBATCH --job-name=${job_name}
#SBATCH --nodes=${nnodes}
#SBATCH --ntasks-per-node=10
##SBATCH --cpus-per-task=1 #
##SBATCH --exclude=pgda206,pgda210,pgda213,pgda214,pgda215,pgda219,pgda220,pgda207,pgda208#,pgda209,pgda218,pgda219,pgda216,pgda217
##SBATCH --share
##SBATCH --mem=90GB # 
#SBATCH --time=150:00:00

source ~/.bashrc

module --ignore_cache load symlinks-1.4-gcc-11.1.0-6t7udsh
module load isis
module load asp/3.3.0 #git-230124

module load py/python
#source ${NOBACKUP}/RING/ring_env/bin/activate

# CHOOSE THE RIGHT ENV FOR YOU!! (you need 755)
#source /explore/nobackup/people/sberton2/RING/ring_env/bin/activate
source /home/tmckenna/nobackup/sfs_clone_env/bin/activate

which python
which stereo_gui

# Set up the nodes list and write to file
scontrol show hostname ${SLURM_NODELIST} | tr ' ' '\n' > tmp_nodeslist_${reg}_${tid}_${sel}_${steps}

python /home/tmckenna/nobackup/new_setup/tests/test_pipeline.py ${tid} ${sel} ${steps}

# Delete the temporary list of nodes
/bin/rm -fv tmp_nodeslist_${reg}_${tid}_${sel}_${steps}
EOF

# Submit the generated script to SLURM
sbatch .tmp_slurm/_slurm_script_${reg}_${tid}_${sel}_${steps}.sh

# Optionally, remove the temporary script after submission
rm .tmp_slurm/_slurm_script_${reg}_${tid}_${sel}_${steps}.sh
