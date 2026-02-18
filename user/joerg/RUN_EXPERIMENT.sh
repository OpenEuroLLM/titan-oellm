


# run on JUWELS interactive node (after srun --pty bash)
TITAN_USER=joerg CLUSTER=juwels DATASET=slimpajama_627b TOKENIZER=neox CONFIG=qwen3_custom.tom bash submit_job.sh --local -- --model.flavor=125M --job.dump_folder=outputs/test_oellm --metrics.save_tb_folder=tb --optimizer.lr=0.0022 --parameter_logging.log-parameters --parameter_logging.log-gradients --training.local_batch_size=8


# submit to JUWELS cluster (SLURM)
TITAN_USER=joerg DATASET=slimpajama_627b TOKENIZER=neox CLUSTER=juwels CONFIG=qwen3_custom.toml bash submit_job.sh --nodes=1 --time=0:15:00 --partition=develbooster -- --model.flavor=125M --job.dump_folder=outputs/test_oellm --metrics.save_tb_folder=tb --optimizer.lr=0.0022 --parameter_logging.log-parameters --parameter_logging.log-gradients --training.local_batch_size=8 --validation.enable --metrics.log_freq=500 --training.seq_len=2048 --training.steps=12000