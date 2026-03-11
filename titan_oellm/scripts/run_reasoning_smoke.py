#!/usr/bin/env python
"""
Quick smoke test for reasoning SFT on a small model + small HF dataset.

Runs a short SFT job using TorchTitan with minimal steps.

Example:
    python titan_oellm/scripts/run_reasoning_smoke.py
    python titan_oellm/scripts/run_reasoning_smoke.py --steps 50 --seq-len 512
"""

import argparse
import sys
from pathlib import Path

import os
import debugpy

# if int(os.environ["RANK"]) == 0:
#     debugpy.listen(("0.0.0.0", 4242))
#     print("Rank 0 waiting for debugger")
#     debugpy.wait_for_client()

# Add torchtitan root first to avoid namespace package shadowing
project_root = Path(__file__).resolve().parents[2]
torchtitan_root = project_root / "torchtitan"
if torchtitan_root.exists():
    sys.path.insert(0, str(torchtitan_root))
sys.path.insert(0, str(project_root))

from titan_oellm.scripts.sft import main_sft


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reasoning SFT smoke test")
    parser.add_argument(
        "--config",
        default="titan_oellm/configs/sft_config.toml",
        help="Path to SFT config",
    )
    parser.add_argument(
        "--dataset",
        default="openai/gsm8k",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--dataset-config",
        default="main",
        help="HuggingFace dataset config name (if required)",
    )
    parser.add_argument(
        "--instruction-format",
        default="reasoning_steps",
        help="Instruction format (reasoning_steps or cot)",
    )
    parser.add_argument("--steps", type=int, default=20, help="Training steps")
    parser.add_argument("--seq-len", type=int, default=2048, help="Sequence length")
    parser.add_argument("--batch-size", type=int, default=16, help="Local batch size")
    parser.add_argument("--global-batch-size", type=int, default=64, help="Global batch size")
    parser.add_argument("--model-name", default="gpt_plus", help="Model name")
    parser.add_argument("--model-flavor", default="0.5B", help="Model flavor")
    
    # Parallelism arguments
    parser.add_argument("--dp-replicate", type=int, default=1, help="Data parallel replicate degree")
    parser.add_argument("--dp-shard", type=int, default=1, help="Data parallel shard degree (FSDP)")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel degree")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallel degree")
    parser.add_argument("--cp", type=int, default=1, help="Context parallel degree")
    parser.add_argument("--ep", type=int, default=1, help="Expert parallel degree")
    parser.add_argument("--etp", type=int, default=1, help="Expert tensor parallel degree")
    
    return parser.parse_args()


def main() -> None:
    args = build_args()

    hf_assets_path = (project_root /"assets" / "gpt2").resolve()

    # Debug: print the path we're using
    print(f"[DEBUG] Setting hf_assets_path to: {hf_assets_path}")
    print(f"[DEBUG] Path exists: {hf_assets_path.exists()}")

    overrides = [
        "--job.config_file",
        args.config,
        "--model.name",
        args.model_name,
        "--model.flavor",
        args.model_flavor,
        "--model.tokenizer_path",
        str(hf_assets_path),
        "--training.dataset",
        "sft_dataset",
        "--training.steps",
        str(args.steps),
        "--training.seq_len",
        str(args.seq_len),
        "--training.local_batch_size",
        str(args.batch_size),
        "--training.global_batch_size",
        str(args.global_batch_size),
        "--data.data_prefix",
        args.dataset,
        "--data.dataset_split",
        "train",
        "--data.instruction_format",
        args.instruction_format,
        "--metrics.log_freq",
        "1",
        "--checkpoint.no-enable",  # Disable checkpointing
        "--validation.no-enable",  # Disable validation
    ]

    if args.dataset_config:
        overrides.extend(["--data.hf_dataset_config", args.dataset_config])

    # Add parallelism overrides
    overrides.extend([
        "--parallelism.data_parallel_replicate_degree",
        str(args.dp_replicate),
        "--parallelism.data_parallel_shard_degree",
        str(args.dp_shard),
        "--parallelism.tensor_parallel_degree",
        str(args.tp),
        "--parallelism.pipeline_parallel_degree",
        str(args.pp),
        "--parallelism.context_parallel_degree",
        str(args.cp),
        "--parallelism.expert_parallel_degree",
        str(args.ep),
        "--parallelism.expert_tensor_parallel_degree",
        str(args.etp),
    ])

    sys.argv = [sys.argv[0]] + overrides
    
    # Debug: print sys.argv to verify overrides
    print(f"[DEBUG] sys.argv = {sys.argv}")
    print(f"[DEBUG] Looking for hf_assets_path in argv...")
    for i, arg in enumerate(sys.argv):
        if "hf_assets_path" in arg.lower():
            print(f"[DEBUG] Found at index {i}: {arg}")
            if i + 1 < len(sys.argv):
                print(f"[DEBUG] Value: {sys.argv[i+1]}")
    
    main_sft()


if __name__ == "__main__":
    main()