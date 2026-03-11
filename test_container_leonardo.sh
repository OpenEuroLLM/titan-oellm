#!/bin/bash
#SBATCH --account=OELLM_prod2026
#SBATCH --error=%j.err
#SBATCH --output=%j.out
#SBATCH --partition=boost_usr_prod
##SBATCH --partition=lrd_all_serial
#SBATCH --job-name=test_container_leonardo
#SBATCH --qos=boost_qos_dbg
##SBATCH --qos=boost_qos_bprod
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=32
#SBATCH --tasks-per-node=1
##SBATCH --exclude=lrdn[1400-3456]
#SBATCH --exclude=lrdn[0181-3456]
#SBATCH --gres=gpu:1

module load cuda/12.1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export NCCL_DEBUG=INFO

srun -l apptainer exec --nv titan_leonardo_0.2.1.sif \
  python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available())'