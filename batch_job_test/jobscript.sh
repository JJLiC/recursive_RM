#!/usr/bin/env bash
#SBATCH -J Recursive_RM       # name of job
#SBATCH -N 1                  # number of nodes
#SBATCH -n 1                  # number of tasks --ntasks=
##SBATCH -c 1                 # CPUs per task (for multithreading?), --cpus-per-task
#SBATCH --mem=8G              # requested memory
## SBATCH -p pfisterlab       # partition, --partition
# SBATCH -p common            # partition, --partition
#SBATCH -t 240:00:00          # maximum time limit
#
# other options:
#
##SBATCH -A C3SE500-12-1    # account name
##SBATCH --mail-user christian.haeger@chalmers.se
##SBATCH -o output-%j.stdout 
##SBATCH -e output.stderr
# ===============================================================#
# the script executes from the launch directory with the user
# environment you had when launching
# ===============================================================#
# $HOME = /dscrhome/username is stored in center storage (network disk?)
# ===============================================================#
tic=$(date +%s)

# ===============================================================#
# start MATLAB
# ===============================================================#
# usage: sbatch --array=1-12 jobscript 

WORKER_ID=$(( $SLURM_ARRAY_TASK_ID * 1 ))
module load Python/3.6.4

python3 batch_worker.py $WORKER_ID
sleep 0.5

wait

toc=$(date +%s)
sec=$(expr $toc - $tic)
min=$(expr $sec / 60)
echo Elapsed time: $sec

# End script
# ===============================================================#  