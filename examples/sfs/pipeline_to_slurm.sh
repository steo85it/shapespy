#!/bin/bash

while getopts c:r:t:i:s:n: flag
do
    case "${flag}" in
        c) config_file=${OPTARG};;
        r) reg=${OPTARG};;
        t) tileid=${OPTARG};;
        i) sel=${OPTARG};;
        s) steps_to_run=${OPTARG};;
        n) nnodes=${OPTARG};;
    esac
done

cwd=$(pwd)
echo $config_file $reg $tileid $sel $steps_to_run $nnodes $cwd

# Create a unique job name using the argument
job_name="sfs_${reg}_${tileid}_sel${sel}"

# Generate the SLURM script with the desired job name
mkdir .tmp_slurm/
cat << EOF > .tmp_slurm/_slurm_script_${reg}_${tileid}_sel${sel}.sh
#!/usr/bin/env bash
#SBATCH --account=j1010
#SBATCH --job-name=${job_name}
#SBATCH --nodes=${nnodes}
#SBATCH --ntasks-per-node=10
##SBATCH --cpus-per-task=1 #
#SBATCH --exclude=pgda202
##SBATCH --share
##SBATCH --mem=90GB # 
#SBATCH --time=7:00:00

source ~/.bashrc

module --ignore_cache load symlinks-1.4-gcc-11.1.0-6t7udsh
module load isis
module load asp/3.4.0 #git-230124
module load gcc/12
#module load python
#source ${NOBACKUP}/RING/ring_env/bin/activate

# CHOOSE THE RIGHT ENV FOR YOU!! (you need 755)
source /explore/nobackup/people/sberton2/RING/ring_env/bin/activate
#source /home/tmckenna/nobackup/new_sfs_env/bin/activate

which python
which stereo_gui
ml list
which gcc

# Set up the nodes list and write to file
scontrol show hostname ${SLURM_NODELIST} | tr ' ' '\n' > tmp_nodeslist_${reg}_${tileid}_sel${sel}

python pipeline_.py --config_file=${config_file} --tileid=${tileid} --sel=${sel} --steps_to_run=${steps_to_run} --nodes_list=${cwd}/tmp_nodeslist_${reg}_${tileid}_sel${sel}

# Delete the temporary list of nodes
/bin/rm -fv tmp_nodeslist_${reg}_${tileid}_sel${sel}
EOF

# Submit the generated script to SLURM
sbatch .tmp_slurm/_slurm_script_${reg}_${tileid}_sel${sel}.sh

# Optionally, remove the temporary script after submission
rm .tmp_slurm/_slurm_script_${reg}_${tileid}_sel${sel}.sh
