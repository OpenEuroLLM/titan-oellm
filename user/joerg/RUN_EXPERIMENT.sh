


# run a local experiment
TITAN_USER=joerg CONFIG=user/joerg/configs/debug.toml bash submit_job.sh --local

# submit to JUWELS cluster (SLURM)
TITAN_USER=joerg CLUSTER=juwels DATASET=slimpajama_627b TOKENIZER=neox CONFIG=user/joerg/configs/debug.toml TOKENIZER=neox bash submit_job.sh

# run on JUWELS interactive node (after srun --pty bash)
TITAN_USER=joerg CLUSTER=juwels DATASET=slimpajama_627b TOKENIZER=neox CONFIG=user/joerg/configs/debug.toml bash submit_job.sh --local --checkpoint.enable

