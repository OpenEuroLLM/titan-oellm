



# submit to CAPELLA cluster
TITAN_USER=joerg DATASET=slimpajama_627b TOKENIZER=neox CLUSTER=capella CONFIG=qwen3_custom.toml bash submit_job.sh \
--nodes=1 \
--time=0:15:00 \
-- \
--model.flavor=125M \
--job.dump_folder=outputs/test_oellm_cap_slimpajama \
--metrics.save_tb_folder=tb \
--optimizer.lr=0.0022 \
--parameter_logging.log-parameters \
--parameter_logging.log-gradients \
--training.local_batch_size=8 \
--validation.enable \
--metrics.log_freq=500 \
--training.seq_len=2048 \
--training.steps=12000


TITAN_USER=joerg DATASET=nemotron_cc TOKENIZER=nemotron CLUSTER=capella CONFIG=qwen3_custom.toml bash submit_job.sh \
--nodes=1 \
--time=0:15:00 \
-- \
--model.flavor=125M \
--job.dump_folder=outputs/test_oellm_cap_nemotron \
--metrics.save_tb_folder=tb \
--optimizer.lr=0.0022 \
--parameter_logging.log-parameters \
--parameter_logging.log-gradients \
--training.local_batch_size=8 \
--validation.enable \
--metrics.log_freq=500 \
--training.seq_len=2048 \
--training.steps=12000