


# run a local experiment

apptainer exec --nv     --bind /tmp/cuda-compat:/usr/local/cuda/compat     --bind /tmp/cuda-lib64:/usr/local/cuda/lib64     --bind /tmp/triton_cache:/workspace/.triton     --env LD_PRELOAD=/usr/local/cuda/compat/lib/libcuda.so.1     --env LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/cuda/lib64     --pwd /opt/titan-sci     --bind $(pwd):/opt/titan-sci     --bind /home/joerg:/home/joerg     titan_juwels_0.2.0.sif     torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=12355       -m torchtitan.train --job.config_file user/joerg/configs/local_default.toml 

OUTPUT_DIR=/opt/titan-oellm
apptainer exec --nv \
      --bind /tmp/cuda-compat:/usr/local/cuda/compat \
      --bind /tmp/cuda-lib64:/usr/local/cuda/lib64 \
      --bind /tmp/triton_cache:/workspace/.triton \
      --env LD_PRELOAD=/usr/local/cuda/compat/lib/libcuda.so.1 \
      --env LIBRARY_PATH=/usr/local/cuda/compat/lib:/usr/local/cuda/lib64 \
      --env OUTPUT_DIR=$OUTPUT_DIR \
      --pwd /opt/titan-oellm \
      --bind $(pwd):/opt/titan-oellm \
      --bind /home/joerg:/home/joerg \
      titan_juwels_0.2.0.sif \
      torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=12355 \
        -m torchtitan.train --job.config_file user/joerg/configs/local_default.toml 