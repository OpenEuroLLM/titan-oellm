



# Start local Qwen3 custom training

TITAN_USER=joerg DATASET=openwebtext TOKENIZER=neox CONFIG=qwen3_custom.toml bash submit_job.sh --local --model.flavor=debugmodel --job.dump_folder=outputs/test_oellm --metrics.save_tb_folder=tb --optimizer.lr=0.0022 --parameter_logging.log-parameters --parameter_logging.log-gradients --training.seq_len=512 --training.local_batch_size=2 --metrics.log_freq=50 --data.dataloader=DeterministicPackedDataset


# Start local GPT+ training

TITAN_USER=joerg DATASET=openwebtext TOKENIZER=neox CONFIG=gpt_plus.toml bash submit_job.sh --local --model.flavor=debugmodel --job.dump_folder=outputs/test_gptplus --metrics.save_tb_folder=tb --optimizer.lr=0.0022 --training.seq_len=512 --training.local_batch_size=2 --metrics.log_freq=50 --data.dataloader=DeterministicPackedDataset
