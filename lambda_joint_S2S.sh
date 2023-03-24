#!/bin/bash

# set project
#SBATCH -A CIA-DAMTP-SL2-GPU

# set partitions
#SBATCH -p ampere

# set max wallclock time
#SBATCH --time=24:00:00

# set name of job
#SBATCH --job-name=CV

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

exper=Lambda
meth=joint
data=DSB2018_n10

for pat in {1..30};
do 
    python /home/fs71765/ngruber/joint_denoising_segmentation/run_S2S.py --output_directory /gpfs/data/fs71765/ngruber/outputs_S2S_new --method $meth --patient $pat --GPU 0 --experiment $exper --lam 0.005 --dataset $data
    python /home/fs71765/ngruber/joint_denoising_segmentation/run_S2S.py --output_directory /gpfs/data/fs71765/ngruber/outputs_S2S_new --method $meth --patient $pat --GPU 0 --experiment $exper --lam 0.01  --dataset $data
    python /home/fs71765/ngruber/joint_denoising_segmentation/run_S2S.py --output_directory /gpfs/data/fs71765/ngruber/outputs_S2S_new --method $meth --patient $pat --GPU 0 --experiment $exper --lam 0.015 --dataset $data
    python /home/fs71765/ngruber/joint_denoising_segmentation/run_S2S.py --output_directory /gpfs/data/fs71765/ngruber/outputs_S2S_new --method $meth --patient $pat --GPU 0 --experiment $exper --lam 0.02 --dataset $data
    python /home/fs71765/ngruber/joint_denoising_segmentation/run_S2S.py --output_directory /gpfs/data/fs71765/ngruber/outputs_S2S_new --method $meth --patient $pat --GPU 0 --experiment $exper --lam 0.025 --dataset $data
    python /home/fs71765/ngruber/joint_denoising_segmentation/run_S2S.py --output_directory /gpfs/data/fs71765/ngruber/outputs_S2S_new --method $meth --patient $pat --GPU 0 --experiment $exper --lam 0.03 --dataset $data
    python /home/fs71765/ngruber/joint_denoising_segmentation/run_S2S.py --output_directory /gpfs/data/fs71765/ngruber/outputs_S2S_new --method $meth --patient $pat --GPU 0 --experiment $exper --lam 0.035 --dataset $data
    python /home/fs71765/ngruber/joint_denoising_segmentation/run_S2S.py --output_directory /gpfs/data/fs71765/ngruber/outputs_S2S_new --method $meth --patient $pat --GPU 0 --experiment $exper --lam 0.04 --dataset $data
done
