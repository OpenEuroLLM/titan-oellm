# Quick start

First set up your local (gitignored) `user/` folder — see the "User configuration"
section in `README.md`:

```bash
cp user/example/cluster_paths.toml.example user/cluster_paths.toml
# edit user/cluster_paths.toml to point at your paths
```

## Start local Qwen3 custom training

```bash
DATASET=openwebtext TOKENIZER=neox CONFIG=qwen3_custom.toml bash submit_job.sh --local --model.flavor=debugmodel --job.dump_folder=outputs/test_oellm --metrics.save_tb_folder=tb --optimizer.lr=0.0022 --parameter_logging.log-parameters --parameter_logging.log-gradients --training.seq_len=512 --training.local_batch_size=2 --metrics.log_freq=50 --data.dataloader=DeterministicPackedDataset
```

## Start local GPT+ training

```bash
DATASET=openwebtext TOKENIZER=neox CONFIG=gpt_plus.toml bash submit_job.sh --local --model.flavor=debugmodel --job.dump_folder=outputs/test_gptplus --metrics.save_tb_folder=tb --optimizer.lr=0.0022 --training.seq_len=512 --training.local_batch_size=2 --metrics.log_freq=50 --data.dataloader=DeterministicPackedDataset
```

## Test the config layer without a GPU

The cluster-path/config logic can be unit-tested on a CPU-only machine (no torch, no container):

```bash
python3 tests/test_cluster_config.py
```

A full training run requires a GPU plus the apptainer container and is not supported on CPU-only machines.
