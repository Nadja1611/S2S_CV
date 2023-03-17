#!/bin/sh
#SBATCH -J lambda_joint
#SBATCH -N 1
#SBATCH --partition=zen3_0512_a100x2 
#SBATCH --qos zen3_0512_a100x2 
#SBATCH -o /gpfs/data/fs71765/ngruber/slurm-%j.out
#SBATCH -e /gpfs/data/fs71765/ngruber/slurm-%j.err
#SBATCH --gres=gpu:2

spack load cuda@11.2.2
spack load /ymsyfrl
spack load miniconda3
source activate jds
for pat in {1..20};
do 
    python /home/fs71765/ngruber/joint_denoising_segmentation/run.py --method cv --patient $pat --GPU 0 --experiment /Lambda --lam 0.0005 &
    python /home/fs71765/ngruber/joint_denoising_segmentation/run.py --method cv --patient $pat --GPU 0 --experiment /Lambda --lam 0.001  
    python /home/fs71765/ngruber/joint_denoising_segmentation/run.py --method cv --patient $pat --GPU 0 --experiment /Lambda --lam 0.002 &
    python /home/fs71765/ngruber/joint_denoising_segmentation/run.py --method cv --patient $pat --GPU 0 --experiment /Lambda --lam 0.003
    python /home/fs71765/ngruber/joint_denoising_segmentation/run.py --method cv --patient $pat --GPU 0 --experiment /Lambda --lam 0.004 &
    python /home/fs71765/ngruber/joint_denoising_segmentation/run.py --method cv --patient $pat --GPU 0 --experiment /Lambda --lam 0.005 
    python /home/fs71765/ngruber/joint_denoising_segmentation/run.py --method cv --patient $pat --GPU 0 --experiment /Lambda --lam 0.006 &
    python /home/fs71765/ngruber/joint_denoising_segmentation/run.py --method cv --patient $pat --GPU 0 --experiment /Lambda --lam 0.0075 
    python /home/fs71765/ngruber/joint_denoising_segmentation/run.py --method cv --patient $pat --GPU 0 --experiment /Lambda --lam 0.01 &
    python /home/fs71765/ngruber/joint_denoising_segmentation/run.py --method cv --patient $pat --GPU 0 --experiment /Lambda --lam 0.0125
    python /home/fs71765/ngruber/joint_denoising_segmentation/run.py --method cv --patient $pat --GPU 0 --experiment /Lambda --lam 0.015 &
    python /home/fs71765/ngruber/joint_denoising_segmentation/run.py --method cv --patient $pat --GPU 0 --experiment /Lambda --lam 0.0175 
    python /home/fs71765/ngruber/joint_denoising_segmentation/run.py --method cv --patient $pat --GPU 0 --experiment /Lambda --lam 0.02 &
    python /home/fs71765/ngruber/joint_denoising_segmentation/run.py --method cv --patient $pat --GPU 0 --experiment /Lambda --lam 0.025
    python /home/fs71765/ngruber/joint_denoising_segmentation/run.py --method cv --patient $pat --GPU 0 --experiment /Lambda --lam 0.03
done

