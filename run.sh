#!/bin/bash

#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=NoisyDECOLLE
#SBATCH --output=logs/icmla/%x_%j.log

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
source /home/stadtmann/miniconda3/etc/profile.d/conda.sh

export LD_LIBRARY_PATH="/home/stadtmann/.local/lib/python3.10/site-packages/nvidia_nccl_cu11-2.14.3-py3.10-linux-x86_64.egg/nvidia/nccl/lib/":"/home/stadtmann/.local/lib/python3.10/site-packages/nvidia_cuda_cupti_cu11-11.7.101-py3.10-linux-x86_64.egg/nvidia/cuda_cupti/lib":"/home/stadtmann/.local/lib/python3.10/site-packages/nvidia_cudnn_cu11-8.5.0.96-py3.10-linux-x86_64.egg/nvidia/cudnn/lib/":$LD_LIBRARY_PATH

conda deactivate
conda activate decolle

#CUDA_VISIBLE_DEVICES=2 python train_lenet_decolle.py --params_file /home/stadtmann/1_Projects/noisysnns/parameters/params_dvsgestures.yml --save_dir icmla/noreg --device cuda:0 --seed $SLURM_JOB_ID
for reg in 0.0001 0.0005 0.001 0.005; do
  #ba activity
  for noise in 1 2 4 6; do
    echo "Background activity: $noise at reg2_l=$reg"
    CUDA_VISIBLE_DEVICES=2 python train_lenet_decolle.py --device cuda:0 --params_file /home/stadtmann/1_Projects/noisysnns/parameters/params_dvsgestures.yml --save_dir icmla/ba_${noise}_reg${reg} --seed $SLURM_JOB_ID --reg2_l $reg $reg $reg $reg --ba_noise $noise
  done
  #hot pixels
  for noise in 0.03 0.06 0.09 0.17 0.27; do
    echo "Hot pixels: $noise at reg2_l=$reg"
    CUDA_VISIBLE_DEVICES=2 python train_lenet_decolle.py --device cuda:0 --params_file /home/stadtmann/1_Projects/noisysnns/parameters/params_dvsgestures.yml --save_dir icmla/hotpixel_${noise}_reg${reg} --seed $SLURM_JOB_ID --reg2_l $reg $reg $reg $reg --hot_pixels $noise
  done
  #mismatch
  for noise in 0.1 0.2 0.3 0.4; do
    echo "Mismatch: $noise at reg2_l=$reg"
    CUDA_VISIBLE_DEVICES=2 python train_lenet_decolle.py --device cuda:0 --params_file /home/stadtmann/1_Projects/noisysnns/parameters/params_dvsgestures.yml --save_dir icmla/mismatch_${noise}_reg${reg} --seed $SLURM_JOB_ID --reg2_l $reg $reg $reg $reg --mismatch $noise
  done
  #spike loss
  for noise in 10 40 60 80; do
    echo "Spike loss: $noise at reg2_l=$reg"
    CUDA_VISIBLE_DEVICES=2 python train_lenet_decolle.py --device cuda:0 --params_file /home/stadtmann/1_Projects/noisysnns/parameters/params_dvsgestures.yml --save_dir icmla/spikeloss_${noise}_reg${reg} --seed $SLURM_JOB_ID --reg2_l $reg $reg $reg $reg --spike_loss $noise
  done
  #quantization
  for noise in 2 3 4 6 8; do
    echo "Quantization: $noise at reg2_l=$reg"
    CUDA_VISIBLE_DEVICES=2 python train_lenet_decolle.py --device cuda:0 --params_file /home/stadtmann/1_Projects/noisysnns/parameters/params_dvsgestures.yml --save_dir icmla/quant_${noise}_reg${reg} --seed $SLURM_JOB_ID --reg2_l $reg $reg $reg $reg --quantise_bits $noise
  done
  #thermal
  for noise in 0.0005 0.001 0.005 0.01; do
    echo "Thermal noise: $noise at reg2_l=$reg"
    CUDA_VISIBLE_DEVICES=2 python train_lenet_decolle.py --device cuda:0 --params_file /home/stadtmann/1_Projects/noisysnns/parameters/params_dvsgestures.yml --save_dir icmla/thermal_${noise}_reg${reg} --seed $SLURM_JOB_ID --reg2_l $reg $reg $reg $reg --thermal_noise $noise
  done
done

conda deactivate
