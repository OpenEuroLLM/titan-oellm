# Copyright (c) Titan-OELLM Custom Components.
# All rights reserved.

"""
Titan-OELLM validator for torchtitan v0.2.0.

This module provides a validator compatible with torchtitan v0.2.0's validation
interface while using titan_oellm's custom validation dataloader.
"""

import json
import math
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Generator, Dict, List, Optional, Set

import torch
import torch.nn as nn
from torch.distributed.pipelining.schedules import _PipelineSchedule

from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.components.loss import LossFunction
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.components.validate import BaseValidator
from torchtitan.config import JobConfig
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.tools import utils
from torchtitan.tools.logging import logger

from titan_oellm.constants import IGNORE_INDEX
from titan_oellm.datasets.sci_dataloader import build_sci_validation_dataloader
from titan_oellm.datasets.dataloader.mmap_dataset import MMapDataset
from titan_oellm.datasets.utils.collator import collate_function_document_eval
from functools import partial


def _compute_spearman_correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Compute Spearman rank correlation coefficient with proper tie handling.

    This implementation uses PyTorch and is equivalent to scipy.stats.spearmanr.
    Spearman correlation measures the monotonic relationship between two variables
    by computing Pearson correlation on the ranks.

    Args:
        x: First variable as a 1D tensor
        y: Second variable as a 1D tensor

    Returns:
        Spearman correlation coefficient between -1.0 and 1.0
    """
    # Convert to float for computation
    x = x.float()
    y = y.float()

    # Compute ranks with proper tie handling (average ranks for ties)
    def rank_data_with_ties(data: torch.Tensor) -> torch.Tensor:
        """Assign ranks with average ranks for ties."""
        # Sort the data and get indices
        sorted_vals, sorted_indices = torch.sort(data)

        # Initialize ranks
        n = len(data)
        ranks = torch.zeros(n, dtype=torch.float32, device=data.device)

        # Assign ranks, handling ties with average ranks
        i = 0
        while i < n:
            # Find extent of ties
            j = i
            while j < n and sorted_vals[j] == sorted_vals[i]:
                j += 1

            # Assign average rank to all tied values
            # Ranks are 1-indexed: positions i to j-1 get average of (i+1) to j
            avg_rank = (i + 1 + j) / 2.0
            for k in range(i, j):
                ranks[sorted_indices[k]] = avg_rank

            i = j

        return ranks

    x_ranked = rank_data_with_ties(x)
    y_ranked = rank_data_with_ties(y)

    # Compute Pearson correlation on ranks
    x_mean = x_ranked.mean()
    y_mean = y_ranked.mean()

    x_centered = x_ranked - x_mean
    y_centered = y_ranked - y_mean

    numerator = (x_centered * y_centered).sum()
    denominator = torch.sqrt((x_centered ** 2).sum() * (y_centered ** 2).sum())

    if denominator == 0:
        return float("nan")

    correlation = numerator / denominator
    return correlation.item()


def _compute_auc_roc(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    """
    Compute AUC-ROC using Mann-Whitney U statistic formulation.

    This implementation uses PyTorch and is equivalent to sklearn's roc_auc_score.
    AUC-ROC measures the probability that a randomly chosen positive example
    has a higher score than a randomly chosen negative example.

    This formulation: AUC = P(score_pos > score_neg) + 0.5 * P(score_pos == score_neg)
    naturally handles tied scores correctly.

    Args:
        y_true: Binary labels (0 or 1) as a 1D tensor
        y_score: Predicted scores (continuous) as a 1D tensor

    Returns:
        AUC-ROC value between 0.0 and 1.0
    """
    # Convert to float for computation
    y_true = y_true.float()
    y_score = y_score.float()

    # Get positive and negative samples
    pos_scores = y_score[y_true == 1]
    neg_scores = y_score[y_true == 0]

    n_pos = len(pos_scores)
    n_neg = len(neg_scores)

    if n_pos == 0 or n_neg == 0:
        # AUC is undefined if only one class present
        return float("nan")

    # Count pairs where positive score > negative score (correct ranking)
    # and pairs where they're equal (ties)
    n_correct = 0.0
    n_tied = 0.0

    for pos_score in pos_scores:
        n_correct += (pos_score > neg_scores).sum().item()
        n_tied += (pos_score == neg_scores).sum().item()

    # AUC formula: (correct pairs + 0.5 * tied pairs) / total pairs
    auc = (n_correct + 0.5 * n_tied) / (n_pos * n_neg)

    return auc


class SciValidator(BaseValidator):
    """
    Titan-OELLM validator using sci_dataloader for validation.

    Args:
        job_config: Job configuration
        dp_world_size: Data parallel world size
        dp_rank: Data parallel rank
        tokenizer: Tokenizer instance
        parallel_dims: Parallel dimensions configuration
        loss_fn: Loss function to use for validation
        validation_context: Context manager for validation
        maybe_enable_amp: Context manager for AMP
        metrics_processor: Metrics processor for logging
        pp_schedule: Pipeline parallel schedule (if PP enabled)
        pp_has_first_stage: Whether this rank has first PP stage
        pp_has_last_stage: Whether this rank has last PP stage
    """

    validation_dataloader: BaseDataLoader

    def __init__(
        self,
        job_config: JobConfig,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: BaseTokenizer,
        parallel_dims: ParallelDims,
        loss_fn: LossFunction,
        validation_context: Generator[None, None, None],
        maybe_enable_amp: Generator[None, None, None],
        metrics_processor: MetricsProcessor | None = None,
        pp_schedule: _PipelineSchedule | None = None,
        pp_has_first_stage: bool | None = None,
        pp_has_last_stage: bool | None = None,
    ):
        super().__init__(job_config)
        self.parallel_dims = parallel_dims
        self.loss_fn = loss_fn
        self.validation_context = validation_context
        self.maybe_enable_amp = maybe_enable_amp
        self.metrics_processor = metrics_processor
        self.pp_schedule = pp_schedule
        self.pp_has_first_stage = pp_has_first_stage
        self.pp_has_last_stage = pp_has_last_stage

        # Cache base dataloader config for consistent reset/benchmarking
        self._base_validation_dataloader_config = {
            "dp_world_size": dp_world_size,
            "dp_rank": dp_rank,
            "tokenizer": tokenizer,
            "job_config": job_config,
        }
        self._validation_sets = self._build_validation_sets()
        self.validation_dataloader = self._validation_sets[0]["dataloader"]

        logger.info(
            f"SciValidator: validation enabled with freq={job_config.validation.freq}, "
            f"datasets={len(self._validation_sets)}"
        )

    def should_validate(self, step: int) -> bool:
        """
        Determine if validation should run at this step.

        Overrides base implementation to include final evaluation check.
        Validation runs at:
        - Step 1 (initial validation)
        - Every N steps (where N = validation.freq)
        - Final training step (to match v0.1.0 behavior)

        Args:
            step: Current training step

        Returns:
            True if validation should run, False otherwise
        """
        return step % self.job_config.validation.freq == 0 or step == self.job_config.training.steps

    def _merge_validation_overrides(self, base_config, override_config):
        if override_config is None or not is_dataclass(override_config):
            return base_config

        overrides = {}
        for f in fields(override_config):
            if f.name == "name":
                continue
            value = getattr(override_config, f.name)
            if value is not None:
                overrides[f.name] = value

        if not overrides:
            return base_config
        return replace(base_config, **overrides)

    def _build_validation_sets(self) -> list[dict]:
        validation = self.job_config.validation
        datasets = getattr(validation, "datasets", [])
        validation_sets = []

        if datasets:
            for idx, dataset_cfg in enumerate(datasets):
                if isinstance(dataset_cfg, dict):
                    try:
                        from titan_sci.configs.sci_job_config import ValidationDataset

                        dataset_cfg = ValidationDataset(**dataset_cfg)
                    except Exception as exc:
                        logger.warning(f"Failed to parse validation dataset override: {exc}")
                dataset_name = getattr(dataset_cfg, "name", "") or f"dataset_{idx}"
                merged_validation = self._merge_validation_overrides(validation, dataset_cfg)
                dataset_job_config = replace(self.job_config, validation=merged_validation)
                dataloader_config = {
                    **self._base_validation_dataloader_config,
                    "job_config": dataset_job_config,
                }
                dataloader = build_sci_validation_dataloader(**dataloader_config)
                validation_sets.append(
                    {
                        "name": dataset_name,
                        "config": dataloader_config,
                        "validation": merged_validation,
                        "dataloader": dataloader,
                    }
                )
        else:
            dataloader = build_sci_validation_dataloader(**self._base_validation_dataloader_config)
            validation_sets.append(
                {
                    "name": "default",
                    "config": self._base_validation_dataloader_config,
                    "validation": validation,
                    "dataloader": dataloader,
                }
            )

        return validation_sets

    def _reset_validation_dataloader(self) -> None:
        """
        Reset the validation dataloader to ensure consistent samples across validation runs.

        This method rebuilds the validation dataloader from scratch to guarantee that
        validation at different steps all process the same set of validation samples
        in the same order. This is crucial for:

        1. Reproducible validation results
        2. Consistent comparison across training steps
        3. Preventing rank divergence due to different validation samples
        4. Deterministic validation behavior
        """
        if not hasattr(self, '_validation_dataloader_config'):
            logger.warning("No cached validation dataloader config found. Validation will proceed without reset.")
            return

        try:
            # Rebuild the dataloader from scratch with the same configuration
            # This ensures all internal state (sample_counter, epoch_counter, RNG) is reset
            self.validation_dataloader = build_sci_validation_dataloader(**self._validation_dataloader_config)
            logger.debug("Validation dataloader reset successfully")
        except Exception as e:
            logger.error(f"Failed to reset validation dataloader: {e}. Validation will proceed with existing dataloader.")
            # Continue with existing dataloader - don't fail validation

    def _collect_gate_metrics(
        self,
        model: nn.Module,
        labels: torch.Tensor,
        correct: torch.Tensor,
        ignore_index: int,
        accumulated_gates: torch.Tensor | None = None,
        max_softmax_probs: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """
        Collect gate statistics from model for ACG, ACG_all, and alpha-dependent gates.
        Also collect softmax confidence metrics for comparison.

        Args:
            model: The model to extract gates from
            labels: Ground truth labels (for masking)
            correct: Boolean tensor of correct predictions (for correlation)
            ignore_index: Index to ignore in labels (padding)
            accumulated_gates: Optional pre-accumulated gate tensor (if None, uses model.last_gate_values)
            max_softmax_probs: Optional tensor of max softmax probabilities (for confidence correlation)

        Returns:
            Dictionary of gate and confidence metrics
        """
        metrics = {}

        # Create mask for valid tokens (non-padding)
        mask_flat = (labels != ignore_index).flatten()

        # Use accumulated gates if provided, otherwise use model's last_gate_values
        # Note: Per-layer gates are only available from model (not accumulated across batches)
        # So we skip per-layer stats when using accumulated gates
        use_accumulated = accumulated_gates is not None

        # ACG_all (Multi-layer Adaptive Confidence Gating)
        if hasattr(model, 'use_acg_all') and model.use_acg_all:
            # Per-layer gates (attention and MLP separately) - only from model
            if not use_accumulated and (hasattr(model, 'last_layer_attn_gates') and
                hasattr(model, 'last_layer_mlp_gates') and
                model.last_layer_attn_gates is not None and
                model.last_layer_mlp_gates is not None):

                for layer_idx, (attn_gate, mlp_gate) in enumerate(
                    zip(model.last_layer_attn_gates, model.last_layer_mlp_gates)
                ):
                    # Process attention gate
                    attn_gate_flat = attn_gate.squeeze(-1).flatten()
                    attn_gate_valid = attn_gate_flat[mask_flat]
                    if attn_gate_valid.numel() > 0:
                        metrics[f'acg/val/layer_{layer_idx}/attn/mean'] = attn_gate_valid.mean().item()
                        metrics[f'acg/val/layer_{layer_idx}/attn/std'] = attn_gate_valid.std().item()

                    # Process MLP gate
                    mlp_gate_flat = mlp_gate.squeeze(-1).flatten()
                    mlp_gate_valid = mlp_gate_flat[mask_flat]
                    if mlp_gate_valid.numel() > 0:
                        metrics[f'acg/val/layer_{layer_idx}/mlp/mean'] = mlp_gate_valid.mean().item()
                        metrics[f'acg/val/layer_{layer_idx}/mlp/std'] = mlp_gate_valid.std().item()

            # Final accumulated gate with correlation
            gate_tensor = accumulated_gates if use_accumulated else (
                model.last_gate_values if hasattr(model, 'last_gate_values') else None
            )

            if gate_tensor is not None:
                gate_flat = gate_tensor.squeeze(-1).flatten()
                correct_flat = correct.flatten()

                gate_valid = gate_flat[mask_flat]
                correct_valid = correct_flat[mask_flat]

                if gate_valid.numel() > 1:
                    # Basic statistics
                    metrics['acg/final/mean'] = gate_valid.mean().item()
                    metrics['acg/final/std'] = gate_valid.std().item()

                    # Compute Pearson correlation between gate and accuracy
                    stacked = torch.stack([gate_valid, correct_valid])
                    pearson_corr = torch.corrcoef(stacked)[0, 1]

                    # Handle NaN correlation (can happen if std is 0)
                    if not torch.isnan(pearson_corr):
                        metrics['acg/final/accuracy_correlation_pearson'] = pearson_corr.item()

                    # Compute AUC-ROC (requires both classes to be present)
                    if len(torch.unique(correct_valid)) > 1:
                        try:
                            auc = _compute_auc_roc(correct_valid, gate_valid)
                            if not math.isnan(auc):
                                metrics['acg/final/accuracy_auc'] = auc
                        except (ValueError, RuntimeError):
                            # Can fail in edge cases
                            pass

                    # Compute Spearman rank correlation
                    try:
                        spearman_corr = _compute_spearman_correlation(gate_valid, correct_valid)
                        if not math.isnan(spearman_corr):
                            metrics['acg/final/accuracy_spearman'] = spearman_corr
                    except (ValueError, RuntimeError):
                        # Can fail in edge cases
                        pass

                    # Diagnostic metrics: mean gate values for correct vs incorrect predictions
                    gate_correct = gate_valid[correct_valid == 1]
                    gate_incorrect = gate_valid[correct_valid == 0]

                    if gate_correct.numel() > 0:
                        metrics['acg/final/mean_gate_correct'] = gate_correct.mean().item()
                    if gate_incorrect.numel() > 0:
                        metrics['acg/final/mean_gate_incorrect'] = gate_incorrect.mean().item()
                    if gate_correct.numel() > 0 and gate_incorrect.numel() > 0:
                        metrics['acg/final/gate_separation'] = (
                            gate_correct.mean().item() - gate_incorrect.mean().item()
                        )

        # Single ACG (Adaptive Confidence Gating)
        elif hasattr(model, 'use_acg') and model.use_acg:
            gate_tensor = accumulated_gates if use_accumulated else (
                model.last_gate_values if hasattr(model, 'last_gate_values') else None
            )

            if gate_tensor is not None:
                gate_flat = gate_tensor.squeeze(-1).flatten()
                correct_flat = correct.flatten()

                gate_valid = gate_flat[mask_flat]
                correct_valid = correct_flat[mask_flat]

                if gate_valid.numel() > 1:
                    # Basic statistics
                    metrics['acg/final/mean'] = gate_valid.mean().item()
                    metrics['acg/final/std'] = gate_valid.std().item()

                    # Compute Pearson correlation between gate and accuracy
                    stacked = torch.stack([gate_valid, correct_valid])
                    pearson_corr = torch.corrcoef(stacked)[0, 1]

                    # Handle NaN correlation (can happen if std is 0)
                    if not torch.isnan(pearson_corr):
                        metrics['acg/final/accuracy_correlation_pearson'] = pearson_corr.item()

                    # Compute AUC-ROC (requires both classes to be present)
                    if len(torch.unique(correct_valid)) > 1:
                        try:
                            auc = _compute_auc_roc(correct_valid, gate_valid)
                            if not math.isnan(auc):
                                metrics['acg/final/accuracy_auc'] = auc
                        except (ValueError, RuntimeError):
                            # Can fail in edge cases
                            pass

                    # Compute Spearman rank correlation
                    try:
                        spearman_corr = _compute_spearman_correlation(gate_valid, correct_valid)
                        if not math.isnan(spearman_corr):
                            metrics['acg/final/accuracy_spearman'] = spearman_corr
                    except (ValueError, RuntimeError):
                        # Can fail in edge cases
                        pass

                    # Diagnostic metrics: mean gate values for correct vs incorrect predictions
                    gate_correct = gate_valid[correct_valid == 1]
                    gate_incorrect = gate_valid[correct_valid == 0]

                    if gate_correct.numel() > 0:
                        metrics['acg/final/mean_gate_correct'] = gate_correct.mean().item()
                    if gate_incorrect.numel() > 0:
                        metrics['acg/final/mean_gate_incorrect'] = gate_incorrect.mean().item()
                    if gate_correct.numel() > 0 and gate_incorrect.numel() > 0:
                        metrics['acg/final/gate_separation'] = (
                            gate_correct.mean().item() - gate_incorrect.mean().item()
                        )

        # Alpha-dependent gates (per-layer residual gating)
        if hasattr(model, 'layers'):
            for layer_id, layer in enumerate(model.layers):
                # Attention residual gate
                if (hasattr(layer, 'attn_update') and
                    hasattr(layer.attn_update, 'alpha_dependent') and
                    layer.attn_update.alpha_dependent and
                    hasattr(layer.attn_update, 'last_gate_values') and
                    layer.attn_update.last_gate_values is not None):

                    gate_flat = layer.attn_update.last_gate_values.squeeze(-1).flatten()
                    gate_valid = gate_flat[mask_flat]
                    if gate_valid.numel() > 0:
                        metrics[f'gating/val_attn_layer_{layer_id}/mean'] = gate_valid.mean().item()
                        metrics[f'gating/val_attn_layer_{layer_id}/std'] = gate_valid.std().item()

                # FFN residual gate
                if (hasattr(layer, 'ffn_update') and
                    hasattr(layer.ffn_update, 'alpha_dependent') and
                    layer.ffn_update.alpha_dependent and
                    hasattr(layer.ffn_update, 'last_gate_values') and
                    layer.ffn_update.last_gate_values is not None):

                    gate_flat = layer.ffn_update.last_gate_values.squeeze(-1).flatten()
                    gate_valid = gate_flat[mask_flat]
                    if gate_valid.numel() > 0:
                        metrics[f'gating/val_ffn_layer_{layer_id}/mean'] = gate_valid.mean().item()
                        metrics[f'gating/val_ffn_layer_{layer_id}/std'] = gate_valid.std().item()

        # Softmax confidence metrics (for comparison with ACG)
        if max_softmax_probs is not None:
            softmax_flat = max_softmax_probs.flatten()
            correct_flat = correct.flatten()

            softmax_valid = softmax_flat[mask_flat]
            correct_valid = correct_flat[mask_flat]

            if softmax_valid.numel() > 1:
                # Basic statistics
                metrics['confidence/val/mean'] = softmax_valid.mean().item()
                metrics['confidence/val/std'] = softmax_valid.std().item()

                # Compute Pearson correlation between confidence and accuracy
                stacked = torch.stack([softmax_valid, correct_valid])
                pearson_corr = torch.corrcoef(stacked)[0, 1]

                # Handle NaN correlation (can happen if std is 0)
                if not torch.isnan(pearson_corr):
                    metrics['confidence/val/accuracy_correlation_pearson'] = pearson_corr.item()

                # Compute AUC-ROC (requires both classes to be present)
                if len(torch.unique(correct_valid)) > 1:
                    try:
                        auc = _compute_auc_roc(correct_valid, softmax_valid)
                        if not math.isnan(auc):
                            metrics['confidence/val/accuracy_auc'] = auc
                    except (ValueError, RuntimeError):
                        # Can fail in edge cases
                        pass

                # Compute Spearman rank correlation
                try:
                    spearman_corr = _compute_spearman_correlation(softmax_valid, correct_valid)
                    if not math.isnan(spearman_corr):
                        metrics['confidence/val/accuracy_spearman'] = spearman_corr
                except (ValueError, RuntimeError):
                    # Can fail in edge cases
                    pass

                # Diagnostic metrics: mean confidence for correct vs incorrect predictions
                conf_correct = softmax_valid[correct_valid == 1]
                conf_incorrect = softmax_valid[correct_valid == 0]

                if conf_correct.numel() > 0:
                    metrics['confidence/val/mean_correct'] = conf_correct.mean().item()
                if conf_incorrect.numel() > 0:
                    metrics['confidence/val/mean_incorrect'] = conf_incorrect.mean().item()
                if conf_correct.numel() > 0 and conf_incorrect.numel() > 0:
                    metrics['confidence/val/separation'] = (
                        conf_correct.mean().item() - conf_incorrect.mean().item()
                    )

        return metrics

    @torch.no_grad()
    def _evaluate_benchmark(
        self,
        model: nn.Module,
        benchmark_name: str,
        benchmark_path: str,
        batch_size: int,
    ) -> dict[str, float]:
        """
        Evaluate a single benchmark dataset using standard LM evaluation.

        For WikiText-2/103: Uses concatenated evaluation (standard method).
        All documents are concatenated and processed in non-overlapping chunks.

        For LAMBADA: Uses per-document evaluation for last-token accuracy.

        Args:
            model: Model to evaluate
            benchmark_name: Name of benchmark (wikitext2, wikitext103, lambada)
            benchmark_path: Path prefix to benchmark bin/idx files
            batch_size: Batch size for evaluation

        Returns:
            Dictionary of metrics for this benchmark
        """
        # LAMBADA requires per-document evaluation for accuracy metric
        if 'lambada' in benchmark_name.lower():
            return self._evaluate_benchmark_document_mode(
                model, benchmark_name, benchmark_path, batch_size
            )

        # WikiText-2/103: Use concatenated evaluation (standard method)
        return self._evaluate_benchmark_concatenated(
            model, benchmark_name, benchmark_path, batch_size
        )

    @torch.no_grad()
    def _evaluate_benchmark_concatenated(
        self,
        model: nn.Module,
        benchmark_name: str,
        benchmark_path: str,
        batch_size: int,
    ) -> dict[str, float]:
        """
        Standard LM evaluation: concatenate all documents and process in chunks.

        This is the standard evaluation method for WikiText and similar benchmarks.
        It provides cross-document context and no padding distortion.
        """
        metrics = {}
        device_type = utils.device_type
        seq_len = self.job_config.training.seq_len
        dp_rank = self._validation_dataloader_config['dp_rank']

        # Load benchmark dataset
        dataset = MMapDataset(
            path_prefix=benchmark_path,
            dp_world_size=1,
            dp_rank=0,
            shuffle=False,
            infinite=False,
            seed=42,
            validate=False,
        )

        # Concatenate all documents into one long sequence
        all_tokens = []
        for sample in dataset:
            tokens = sample['tokens']
            if hasattr(tokens, 'tolist'):
                all_tokens.extend(tokens.tolist())
            else:
                all_tokens.extend(list(tokens))

        all_tokens = torch.tensor(all_tokens, dtype=torch.long)
        total_tokens_in_dataset = len(all_tokens)

        if dp_rank == 0:
            logger.info(f"Benchmark {benchmark_name}: loaded {total_tokens_in_dataset:,} tokens")

        # Process in non-overlapping chunks
        total_loss = 0.0
        total_tokens = 0

        # Need seq_len+1 tokens for each chunk (input + 1 target)
        num_chunks = (len(all_tokens) - 1) // seq_len

        for chunk_idx in range(num_chunks):
            start = chunk_idx * seq_len
            end = start + seq_len + 1

            chunk = all_tokens[start:end]
            inputs = chunk[:-1].unsqueeze(0).to(device_type)  # [1, seq_len]
            labels = chunk[1:].unsqueeze(0).to(device_type)   # [1, seq_len]

            with self.maybe_enable_amp:
                pred = model(inputs)
                loss = torch.nn.functional.cross_entropy(
                    pred.flatten(0, 1).float(),
                    labels.flatten(0, 1),
                )

                # All tokens are valid (no padding in concatenated mode)
                chunk_tokens = seq_len
                total_loss += loss.item() * chunk_tokens
                total_tokens += chunk_tokens

        # Compute metrics
        if total_tokens > 0:
            avg_loss = total_loss / total_tokens
            perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
            metrics[f'benchmark/{benchmark_name}/perplexity'] = perplexity
            metrics[f'benchmark/{benchmark_name}/loss'] = avg_loss

        if dp_rank == 0:
            logger.info(f"Benchmark {benchmark_name}: perplexity={perplexity:.3f}, "
                       f"tokens={total_tokens:,}/{total_tokens_in_dataset:,}")

        return metrics

    @torch.no_grad()
    def _evaluate_benchmark_document_mode(
        self,
        model: nn.Module,
        benchmark_name: str,
        benchmark_path: str,
        batch_size: int,
    ) -> dict[str, float]:
        """
        Per-document evaluation for LAMBADA (last-token accuracy metric).

        Each document is evaluated independently, which is required for
        LAMBADA's last-token prediction accuracy metric.
        """
        metrics = {}
        device_type = utils.device_type
        seq_len = self.job_config.training.seq_len
        dp_rank = self._validation_dataloader_config['dp_rank']

        # Load benchmark dataset
        dataset = MMapDataset(
            path_prefix=benchmark_path,
            dp_world_size=1,
            dp_rank=0,
            shuffle=False,
            infinite=False,
            seed=42,
            validate=False,
        )

        # Create collate function for document mode
        collate_fn = partial(
            collate_function_document_eval,
            seq_len=seq_len,
            ignore_index=IGNORE_INDEX,
            pad_id=0,
        )

        # Create simple dataloader
        from torch.utils.data import DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=0,
        )

        total_loss = 0.0
        total_tokens = 0
        total_correct_last = 0
        total_docs = 0

        for batch_idx, (input_dict, labels) in enumerate(dataloader):
            inputs = input_dict["input"].to(device_type)
            labels = labels.to(device_type)

            with self.maybe_enable_amp:
                pred = model(inputs)
                loss = torch.nn.functional.cross_entropy(
                    pred.flatten(0, 1).float(),
                    labels.flatten(0, 1),
                    ignore_index=IGNORE_INDEX,
                )

                non_padding_mask = (labels != IGNORE_INDEX)
                batch_tokens = non_padding_mask.sum().item()

                total_loss += loss.item() * batch_tokens
                total_tokens += batch_tokens

                # LAMBADA: accuracy on last token of each document
                for i in range(inputs.shape[0]):
                    doc_mask = non_padding_mask[i]
                    if doc_mask.any():
                        last_pos = doc_mask.nonzero()[-1].item()
                        pred_token = torch.argmax(pred[i, last_pos])
                        target_token = labels[i, last_pos]
                        if pred_token == target_token:
                            total_correct_last += 1
                        total_docs += 1

        # Compute metrics
        if total_tokens > 0:
            avg_loss = total_loss / total_tokens
            perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
            metrics[f'benchmark/{benchmark_name}/perplexity'] = perplexity
            metrics[f'benchmark/{benchmark_name}/loss'] = avg_loss

        if total_docs > 0:
            accuracy = total_correct_last / total_docs
            metrics[f'benchmark/{benchmark_name}/accuracy'] = accuracy

        if dp_rank == 0:
            logger.info(f"Benchmark {benchmark_name}: perplexity={metrics.get(f'benchmark/{benchmark_name}/perplexity', 'N/A'):.3f}, "
                       f"accuracy={metrics.get(f'benchmark/{benchmark_name}/accuracy', 'N/A'):.4f}, "
                       f"tokens={total_tokens}, docs={total_docs}")

        return metrics

    @torch.no_grad()
    def _evaluate_all_benchmarks(
        self,
        model: nn.Module,
    ) -> dict[str, float]:
        """
        Evaluate all configured benchmark datasets.

        Args:
            model: Model to evaluate

        Returns:
            Dictionary of all benchmark metrics
        """
        metrics = {}

        # Check if benchmarks are enabled
        if not hasattr(self.job_config, 'benchmarks') or not self.job_config.benchmarks.enable:
            return metrics

        benchmarks_config = self.job_config.benchmarks
        batch_size = benchmarks_config.batch_size

        # Evaluate each configured benchmark
        benchmark_configs = [
            ('wikitext2', benchmarks_config.wikitext2_path),
            ('wikitext103', benchmarks_config.wikitext103_path),
            ('lambada', benchmarks_config.lambada_path),
        ]

        for name, path in benchmark_configs:
            if path and path.strip():
                benchmark_metrics = self._evaluate_benchmark(
                    model=model,
                    benchmark_name=name,
                    benchmark_path=path,
                    batch_size=batch_size,
                )
                metrics.update(benchmark_metrics)

        return metrics

    @torch.no_grad()
    def validate(
        self,
        model_parts: list[nn.Module],
        step: int,
    ) -> None:
        """
        Perform validation evaluation.

        Args:
            model_parts: List of model parts (1 for non-PP, multiple for PP)
            step: Current training step
        """
        # Reset validation dataloader to ensure consistent samples across validation runs
        self._reset_validation_dataloader()

        # Set model to eval mode
        for model in model_parts:
            model.eval()

        parallel_dims = self.parallel_dims
        device_type = utils.device_type

        accumulated_losses = []
        total_tokens = 0
        num_steps = 0
        total_correct = 0
        total_tokens_for_accuracy = 0

        # Store last batch info for gate metrics (to avoid memory accumulation)
        last_batch_correct = None
        last_batch_labels = None
        last_batch_max_softmax = None

        # Determine max validation steps
        max_eval_samples = self.job_config.validation.max_eval_samples
        if max_eval_samples == -1:
            # Process entire validation set
            max_steps = float('inf')
        else:
            # Estimate max steps based on max_eval_samples
            # This is approximate since actual samples per batch may vary
            max_steps = max_eval_samples

        # Use iterator to better handle dataloader exhaustion
        dataloader_iter = iter(self.validation_dataloader)

        while True:
            # Check if we've reached max steps
            if num_steps >= max_steps:
                break

            # Try to get next batch
            try:
                input_dict, labels = next(dataloader_iter)
                has_data = True
            except StopIteration:
                has_data = False

            # Synchronize whether all ranks have data to prevent deadlocks
            # This ensures all ranks exit the loop together
            has_data_tensor = torch.tensor(
                [1 if has_data else 0],
                dtype=torch.int32,
                device=device_type
            )

            if parallel_dims.dp_cp_enabled:
                import torch.distributed as dist
                # Use MIN so if any rank is out of data, all ranks stop
                dist.all_reduce(
                    has_data_tensor,
                    op=dist.ReduceOp.MIN,
                    group=parallel_dims.get_optional_mesh("loss").get_group()
                )

            # If any rank ran out of data, stop validation
            if has_data_tensor.item() == 0:
                break

            # Track tokens for metrics
            if self.metrics_processor:
                self.metrics_processor.ntokens_since_last_log += labels.numel()

            # Move tensors to device
            for k, v in input_dict.items():
                if isinstance(v, torch.Tensor):
                    input_dict[k] = v.to(device_type)
            inputs = input_dict["input"]
            labels = labels.to(device_type)

            # Setup context parallel if enabled
            optional_context_parallel_ctx = (
                dist_utils.create_context_parallel_ctx(
                    cp_mesh=parallel_dims.world_mesh["cp"],
                    cp_buffers=[inputs, labels] + [m.freqs_cis for m in model_parts],
                    cp_seq_dims=[1, 1] + [0 for _ in model_parts],
                    cp_no_restore_buffers={inputs, labels},
                    cp_rotate_method=self.job_config.parallelism.context_parallel_rotate_method,
                )
                if parallel_dims.cp_enabled
                else None
            )

            if parallel_dims.pp_enabled:
                # Pipeline parallel validation
                assert self.pp_schedule is not None
                assert self.pp_has_first_stage is not None
                assert self.pp_has_last_stage is not None

                with self.validation_context(optional_context_parallel_ctx):
                    targets, losses = (
                        (labels, []) if self.pp_has_last_stage else (None, None)
                    )
                    if self.pp_has_first_stage:
                        self.pp_schedule.eval(
                            inputs,
                            target=targets,
                            losses=losses,
                        )
                    else:
                        self.pp_schedule.eval(target=targets, losses=losses)

                # Accumulate losses across pipeline microbatches
                if self.pp_has_last_stage and losses:
                    loss = torch.mean(torch.stack(losses))
                    accumulated_losses.append(loss)
                    # For PP, we can't easily compute accuracy without gathering logits
            else:
                # Non-PP validation
                with self.validation_context(optional_context_parallel_ctx):
                    assert len(model_parts) == 1
                    with self.maybe_enable_amp:
                        # Extract extra inputs (cu_seqlens, max_seqlen for Flash Attention)
                        extra_inputs = {k: v for k, v in input_dict.items() if k != "input"}

                        # Handle flex attention masks if enabled
                        extra_kwargs = {}
                        model_args = getattr(model_parts[0], 'model_args', None)
                        if model_args is not None and getattr(model_args, 'use_flex_attn', False):
                            tokenizer = self._validation_dataloader_config.get('tokenizer')
                            extra_kwargs["attention_masks"] = model_parts[0].get_attention_masks(
                                input_batch=inputs,
                                tokenizer=tokenizer,
                                extra_inputs={},
                            )
                        pred = model_parts[0](inputs, **extra_inputs, **extra_kwargs)
                        # Use cross_entropy with ignore_index for proper validation loss
                        # (training loss_fn doesn't use ignore_index, which would corrupt metrics)
                        loss = torch.nn.functional.cross_entropy(
                            pred.flatten(0, 1).float(),
                            labels.flatten(0, 1),
                            ignore_index=IGNORE_INDEX,
                        )
                        accumulated_losses.append(loss)

                        # Compute accuracy
                        pred_tokens = torch.argmax(pred, dim=-1)
                        correct = (pred_tokens == labels).float()

                        # Extract max softmax probabilities for confidence metrics
                        pred_probs = torch.softmax(pred, dim=-1)  # Convert logits to probabilities
                        max_softmax_probs = torch.max(pred_probs, dim=-1).values  # Max prob per token

                        ignore_index = getattr(self.validation_dataloader, 'ignore_index', IGNORE_INDEX)
                        non_padding_mask = (labels != ignore_index)
                        correct_masked = correct * non_padding_mask
                        total_correct += correct_masked.sum().item()
                        total_tokens_for_accuracy += non_padding_mask.sum().item()

                        # Track tokens
                        total_tokens += non_padding_mask.sum().item()

                        # Store last batch for gate metrics (model's last_gate_values are from this batch)
                        # Only keep the last batch to avoid memory accumulation
                        last_batch_correct = correct
                        last_batch_labels = labels
                        last_batch_max_softmax = max_softmax_probs

            num_steps += 1

        # Set model back to train mode
        for model in model_parts:
            model.train()

        # Compute validation metrics
        if len(accumulated_losses) > 0:
            avg_val_loss = torch.mean(torch.stack(accumulated_losses))

            # Reduce across DP ranks if needed
            if parallel_dims.dp_cp_enabled:
                avg_val_loss = dist_utils.dist_mean(
                    avg_val_loss, parallel_dims.get_optional_mesh("loss")
                )
                # dist_mean already returns a float
                val_loss = avg_val_loss
            else:
                # avg_val_loss is a tensor, need to call .item()
                val_loss = avg_val_loss.item()
            val_perplexity = math.exp(val_loss) if val_loss < 100 else float('inf')

            # Compute global accuracy if available
            if total_tokens_for_accuracy > 0:
                val_accuracy = total_correct / total_tokens_for_accuracy
                if parallel_dims.dp_cp_enabled:
                    import torch.distributed as dist
                    # Reduce accuracy across DP ranks
                    accuracy_tensor = torch.tensor(val_accuracy, device=device_type)
                    dist.all_reduce(accuracy_tensor, op=dist.ReduceOp.AVG, group=parallel_dims.get_optional_mesh("loss").get_group())
                    val_accuracy = accuracy_tensor.item()
            else:
                val_accuracy = 0.0

            # Log validation metrics
            color = self.metrics_processor.color if self.metrics_processor else None
            if color:
                log_msg = (
                    f"{color.blue}step: {step:>6} "
                    f"{color.green}val_loss: {val_loss:.6f} "
                    f"{color.green}val_perplexity: {val_perplexity:.3f} "
                )
                if total_tokens_for_accuracy > 0:
                    log_msg += f"{color.green}val_accuracy: {val_accuracy:.4f} "
                log_msg += (
                    f"{color.yellow}val_tokens: {total_tokens:>6} "
                    f"{color.yellow}val_steps: {num_steps:>3}{color.reset}"
                )
                logger.info(log_msg)
            else:
                logger.info(
                    f"step: {step:>6} val_loss: {val_loss:.6f} "
                    f"val_perplexity: {val_perplexity:.3f} "
                    f"val_tokens: {total_tokens:>6} val_steps: {num_steps:>3}"
                )

            # Collect gate metrics (ACG, ACG_all, alpha-dependent) if available
            # Use last batch only to avoid memory accumulation and sync issues
            gate_metrics = {}
            if (not parallel_dims.pp_enabled and len(model_parts) == 1 and
                last_batch_correct is not None and last_batch_labels is not None):
                # Only collect gate metrics for non-PP mode
                # For PP mode, we can't easily access intermediate gate values
                # Use last batch only (model's last_gate_values correspond to this batch)
                ignore_index = getattr(self.validation_dataloader, 'ignore_index', IGNORE_INDEX)
                gate_metrics = self._collect_gate_metrics(
                    model_parts[0],
                    last_batch_labels,
                    last_batch_correct,
                    ignore_index,
                    accumulated_gates=None,  # Use model's last_gate_values
                    max_softmax_probs=last_batch_max_softmax  # Pass softmax confidence values
                )

            # Evaluate standard benchmarks if configured (non-PP only)
            benchmark_metrics = {}
            if not parallel_dims.pp_enabled and len(model_parts) == 1:
                benchmark_metrics = self._evaluate_all_benchmarks(model_parts[0])

            # Log to metrics processor (TensorBoard/WandB)
            if self.metrics_processor and hasattr(self.metrics_processor, 'logger') and self.metrics_processor.logger:
                validation_metrics = {
                    "validation/val_loss": val_loss,
                    "validation/val_perplexity": val_perplexity,
                }
                if total_tokens_for_accuracy > 0:
                    validation_metrics["validation/val_accuracy"] = val_accuracy

                # Add gate metrics
                validation_metrics.update(gate_metrics)

                # Add benchmark metrics
                validation_metrics.update(benchmark_metrics)

                self.metrics_processor.logger.log(validation_metrics, step)


def build_sci_validator(
    job_config: JobConfig,
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    parallel_dims: ParallelDims,
    loss_fn: LossFunction,
    validation_context: Generator[None, None, None],
    maybe_enable_amp: Generator[None, None, None],
    metrics_processor: MetricsProcessor | None = None,
    pp_schedule: _PipelineSchedule | None = None,
    pp_has_first_stage: bool | None = None,
    pp_has_last_stage: bool | None = None,
) -> BaseValidator:
    """
    Build a SciValidator for titan_oellm validation.

    This is compatible with torchtitan v0.2.0's build_validator signature
    while using titan_oellm's custom validation dataloader.

    Args:
        job_config: Job configuration
        dp_world_size: Data parallel world size
        dp_rank: Data parallel rank
        tokenizer: Tokenizer instance
        parallel_dims: Parallel dimensions configuration
        loss_fn: Loss function to use for validation
        validation_context: Context manager for validation
        maybe_enable_amp: Context manager for AMP
        metrics_processor: Metrics processor for logging
        pp_schedule: Pipeline parallel schedule (if PP enabled)
        pp_has_first_stage: Whether this rank has first PP stage
        pp_has_last_stage: Whether this rank has last PP stage

    Returns:
        SciValidator instance
    """
    return SciValidator(
        job_config=job_config,
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        tokenizer=tokenizer,
        parallel_dims=parallel_dims,
        loss_fn=loss_fn,
        validation_context=validation_context,
        maybe_enable_amp=maybe_enable_amp,
        metrics_processor=metrics_processor,
        pp_schedule=pp_schedule,
        pp_has_first_stage=pp_has_first_stage,
        pp_has_last_stage=pp_has_last_stage,
    )
