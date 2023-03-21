#!/bin/bash

# set project
#SBATCH -A CIA-DAMTP-SL2-GPU

# set partitions
#SBATCH -p ampere

# set max wallclock time
#SBATCH --time=3:30:00

# set name of job
#SBATCH --job-name=CV_only

# set the number of nodes
#SBATCH --nodes=1

# set number of GPUs
#SBATCH --gres=gpu:1

# set number of CPUs
#SBATCH -c 32

# set size of Memory/RAM
#SBATCH --mem=250G

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=nadjagruber2@gmail.com

#! sbatch directives end here (put any additional directives above this line)

#! Notes:
#! Charging is determined by GPU number*walltime.


#! ############################################################
#! Modify the settings below to specify the application's environment, location
#! and launch method:

# output server
#SBATCH -o cv2030.txt

# error server
#SBATCH -e cv_e2030.txt
#! Optionally modify the environment seen by the application
#! (note that SLURM reproduces the environment at submission irrespective of ~/.bashrc):
. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge    
module load miniconda/3
#module load cuda/11.2
#module load cudnn/8.1_cuda-11.2             # REQUIRED - loads the basic environment
module load cuda/11.4
#! Insert additional module load commands after this line if needed:

source activate pytorch

for pat in {1..3800};
do 
    python /home/ng529/rds/hpc-work/code/S2S_CV/S2S_CV/run_S2S.py --method cv --patient $pat --GPU 0 --experiment /Lambda --lam 0.0005 &
    python /home/ng529/rds/hpc-work/code/S2S_CV/S2S_CV/run_S2S.py --method cv --patient $pat --GPU 0 --experiment /Lambda --lam 0.001  
    python /home/ng529/rds/hpc-work/code/S2S_CV/S2S_CV/run_S2S.py --method cv --patient $pat --GPU 0 --experiment /Lambda --lam 0.002 &
    python /home/ng529/rds/hpc-work/code/S2S_CV/S2S_CV/run_S2S.py --method cv --patient $pat --GPU 0 --experiment /Lambda --lam 0.003
    python /home/ng529/rds/hpc-work/code/S2S_CV/S2S_CV/run_S2S.py --method cv --patient $pat --GPU 0 --experiment /Lambda --lam 0.004 &
    python /home/ng529/rds/hpc-work/code/S2S_CV/S2S_CV/run_S2S.py --method cv --patient $pat --GPU 0 --experiment /Lambda --lam 0.005 
    python /home/ng529/rds/hpc-work/code/S2S_CV/S2S_CV/run_S2S.py --method cv --patient $pat --GPU 0 --experiment /Lambda --lam 0.006 &
    python /home/ng529/rds/hpc-work/code/S2S_CV/S2S_CV/run_S2S.py --method cv --patient $pat --GPU 0 --experiment /Lambda --lam 0.0075 
    python /home/ng529/rds/hpc-work/code/S2S_CV/S2S_CV/run_S2S.py --method cv --patient $pat --GPU 0 --experiment /Lambda --lam 0.01 &
    python /home/ng529/rds/hpc-work/code/S2S_CV/S2S_CV/run_S2S.py --method cv --patient $pat --GPU 0 --experiment /Lambda --lam 0.0125
    python /home/ng529/rds/hpc-work/code/S2S_CV/S2S_CV/run_S2S.py --method cv --patient $pat --GPU 0 --experiment /Lambda --lam 0.015 &
    python /home/ng529/rds/hpc-work/code/S2S_CV/S2S_CV/run_S2S.py --method cv --patient $pat --GPU 0 --experiment /Lambda --lam 0.0175 
    python /home/ng529/rds/hpc-work/code/S2S_CV/S2S_CV/run_S2S.py --method cv --patient $pat --GPU 0 --experiment /Lambda --lam 0.02 &
    python /home/ng529/rds/hpc-work/code/S2S_CV/S2S_CV/run_S2S.py --method cv --patient $pat --GPU 0 --experiment /Lambda --lam 0.025
    python /home/ng529/rds/hpc-work/code/S2S_CV/S2S_CV/run_S2S.py --method cv --patient $pat --GPU 0 --experiment /Lambda --lam 0.03
done

