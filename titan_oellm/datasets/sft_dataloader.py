"""
SFT DataLoader for instruction-formatted and reasoning datasets.

Supports:
- HuggingFace datasets (any dataset with text/conversation data)
- Local JSONL files
- Instruction formats: Alpaca, ChatML, ShareGPT, etc.
- Reasoning formats: Chain-of-Thought, step-by-step reasoning
- Auto-detection of dataset structure

The dataloader formats the data and creates loss masks to only compute
loss on the response tokens, not the instruction tokens.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Union

import torch
from torch.utils.data import Dataset

from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import JobConfig

from titan_oellm.constants import IGNORE_INDEX

logger = logging.getLogger(__name__)


def detect_format(example: dict) -> str:
    """Auto-detect the instruction format from a dataset example."""
    # Check for reasoning formats first (more specific)
    if "steps" in example or "reasoning" in example:
        return "reasoning_steps"
    elif ("chain_of_thought" in example or "cot" in example or 
          "thought" in example or "thinking" in example):
        return "cot"
    elif "problem" in example and ("solution" in example or "answer" in example):
        return "problem_solution"
    
    # Check for conversation/instruction formats
    if "messages" in example:
        return "chatml"
    elif "conversations" in example:
        return "sharegpt"
    elif "instruction" in example and "output" in example:
        return "alpaca"
    elif "prompt" in example and "completion" in example:
        return "prompt_completion"
    elif "text" in example:
        return "text"
    else:
        # Try to find any text-like fields
        for key in ["question", "query", "input"]:
            if key in example:
                return "auto"
        return "unknown"


class InstructionDataset(Dataset):
    """
    Dataset for instruction-formatted data used in supervised fine-tuning.
    
    Supports:
    - HuggingFace datasets (load via dataset name/path)
    - Local JSONL files
    - Auto-detection of format
    - Multiple instruction formats
    """
    
    def __init__(
        self,
        data_source: Union[str, object],
        tokenizer: BaseTokenizer,
        seq_len: int,
        instruction_format: str = "auto",
        seed: int = 42,
        split: str = "train",
        hf_dataset_name: Optional[str] = None,
        hf_dataset_config: Optional[str] = None,
        text_field: Optional[str] = None,
    ):
        """
        Args:
            data_source: Path to JSONL file, HF dataset name, or dataset object
            tokenizer: Tokenizer to use
            seq_len: Maximum sequence length
            instruction_format: Format (alpaca, chatml, sharegpt, auto)
            seed: Random seed for shuffling
            split: Dataset split to use (for HF datasets)
            hf_dataset_name: Explicitly specify HF dataset name
            text_field: Field containing text data (for simple datasets)
        """
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.instruction_format = instruction_format
        self.text_field = text_field
        
        # Load data from various sources
        self.data = []
        
        # Try to load as HuggingFace dataset first
        if hf_dataset_name or (isinstance(data_source, str) and not Path(data_source).exists()):
            try:
                from datasets import load_dataset
                
                dataset_name = hf_dataset_name or data_source
                if hf_dataset_config:
                    logger.info(
                        f"Loading HuggingFace dataset: {dataset_name}/{hf_dataset_config}, split: {split}"
                    )
                    dataset = load_dataset(dataset_name, hf_dataset_config, split=split)
                else:
                    logger.info(f"Loading HuggingFace dataset: {dataset_name}, split: {split}")
                    dataset = load_dataset(dataset_name, split=split)
                self.data = list(dataset)
                
                logger.info(f"Loaded {len(self.data)} examples from HuggingFace dataset")
                
                # Auto-detect format from first example
                if self.instruction_format == "auto" and len(self.data) > 0:
                    self.instruction_format = detect_format(self.data[0])
                    logger.info(f"Auto-detected format: {self.instruction_format}")
                    
            except Exception as e:
                logger.warning(f"Failed to load as HuggingFace dataset: {e}")
                logger.info("Attempting to load as local file...")
        
        # Load from local JSONL file if not already loaded
        if not self.data and isinstance(data_source, str):
            data_file = Path(data_source)
            
            if not data_file.exists():
                raise FileNotFoundError(
                    f"Data file not found: {data_source}\n"
                    f"Tried as HuggingFace dataset and local file."
                )
            
            logger.info(f"Loading instruction data from {data_source}")
            
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        self.data.append(json.loads(line))
            
            logger.info(f"Loaded {len(self.data)} instruction examples from file")
            
            # Auto-detect format
            if self.instruction_format == "auto" and len(self.data) > 0:
                self.instruction_format = detect_format(self.data[0])
                logger.info(f"Auto-detected format: {self.instruction_format}")
        
        if not self.data:
            raise ValueError("No data loaded from any source")
        
        # Shuffle with seed for reproducibility
        import random
        random.seed(seed)
        random.shuffle(self.data)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def format_alpaca(self, example: dict) -> tuple[str, str]:
        """
        Format Alpaca-style instruction.
        
        Format:
            Below is an instruction that describes a task...
            ### Instruction: {instruction}
            ### Input: {input}
            ### Response: {output}
        """
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output = example.get("output", "")
        
        if input_text:
            prompt = (
                f"Below is an instruction that describes a task, paired with an input that provides further context. "
                f"Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{input_text}\n\n"
                f"### Response:\n"
            )
        else:
            prompt = (
                f"Below is an instruction that describes a task. "
                f"Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{instruction}\n\n"
                f"### Response:\n"
            )
        
        return prompt, output
    
    def format_chatml(self, example: dict) -> tuple[str, str]:
        """
        Format ChatML-style conversation.
        
        Format:
            <|im_start|>system
            {system}<|im_end|>
            <|im_start|>user
            {user}<|im_end|>
            <|im_start|>assistant
            {assistant}<|im_end|>
        """
        messages = example.get("messages", [])
        
        prompt_parts = []
        response = ""
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "assistant":
                response = content
                break
            else:
                prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        
        prompt_parts.append("<|im_start|>assistant\n")
        prompt = "\n".join(prompt_parts)
        
        return prompt, response
    
    def format_sharegpt(self, example: dict) -> tuple[str, str]:
        """
        Format ShareGPT-style conversation.
        
        Format from conversations with 'from' and 'value' keys.
        """
        conversations = example.get("conversations", [])
        
        prompt_parts = []
        response = ""
        
        for conv in conversations:
            role = conv.get("from", "human")
            content = conv.get("value", "")
            
            if role in ["gpt", "assistant"]:
                response = content
                break
            elif role in ["human", "user"]:
                prompt_parts.append(f"User: {content}")
            elif role == "system":
                prompt_parts.append(f"System: {content}")
        
        prompt_parts.append("Assistant:")
        prompt = "\n".join(prompt_parts)
        
        return prompt, response
    
    def format_prompt_completion(self, example: dict) -> tuple[str, str]:
        """Format simple prompt-completion pairs."""
        prompt = example.get("prompt", "")
        completion = example.get("completion", "")
        return prompt, completion
    
    def format_text(self, example: dict) -> tuple[str, str]:
        """
        Format plain text data by using text_field or 'text' key.
        For plain text, we don't mask anything (train on full text).
        """
        if self.text_field:
            text = example.get(self.text_field, "")
        else:
            text = example.get("text", "")
        
        # For plain text, we split arbitrarily or use whole text
        # Here we use empty prompt and full text as response
        return "", text
    
    def format_auto(self, example: dict) -> tuple[str, str]:
        """
        Auto-detect and format based on available fields.
        Tries common field names for question/answer pairs.
        """
        # Try common prompt field names
        prompt_fields = ["question", "query", "prompt", "input", "instruction"]
        response_fields = ["answer", "response", "output", "completion", "text"]
        
        prompt = ""
        response = ""
        
        for field in prompt_fields:
            if field in example and example[field]:
                prompt = str(example[field])
                break
        
        for field in response_fields:
            if field in example and example[field]:
                response = str(example[field])
                break
        
        if not response:
            # Fallback: use any text-like field
            for key, value in example.items():
                if isinstance(value, str) and value:
                    response = value
                    break
        
        return prompt, response
    
    def format_cot(self, example: dict) -> tuple[str, str]:
        """
        Format Chain-of-Thought (CoT) datasets.
        
        Combines problem/question with chain-of-thought reasoning.
        Common field names:
        - Problem: question, problem, query, prompt
        - CoT: chain_of_thought, thought, thinking, reasoning, explanation
        - Answer: answer, response, output, solution, final_answer
        """
        # Find problem
        problem_fields = ["question", "problem", "query", "prompt", "input", "instruction"]
        problem = ""
        for field in problem_fields:
            if field in example and example[field]:
                problem = str(example[field])
                break
        
        # Find chain-of-thought
        cot_fields = ["chain_of_thought", "thought", "thinking", "reasoning", "explanation", "steps"]
        cot = ""
        for field in cot_fields:
            if field in example and example[field]:
                cot_value = example[field]
                if isinstance(cot_value, list):
                    # Handle list of reasoning steps
                    cot = "\n".join([f"Step {i+1}: {step}" for i, step in enumerate(cot_value)])
                else:
                    cot = str(cot_value)
                break
        
        # Find answer
        answer_fields = ["answer", "response", "output", "solution", "final_answer", "conclusion"]
        answer = ""
        for field in answer_fields:
            if field in example and example[field]:
                answer = str(example[field])
                break
        
        # Combine: problem is prompt, cot + answer is response
        prompt = problem
        response = f"{cot}\n\nFinal Answer: {answer}" if cot and answer else answer
        
        return prompt, response
    
    def format_reasoning_steps(self, example: dict) -> tuple[str, str]:
        """
        Format multi-step reasoning datasets.
        
        Structure: problem -> intermediate reasoning steps -> final answer
        Common field names:
        - Problem: question, problem, query, context
        - Steps: steps, reasoning, intermediate_steps, process
        - Answer: answer, final_answer, solution, output
        """
        # Find problem/question
        problem_fields = ["question", "problem", "query", "context", "prompt", "input"]
        problem = ""
        for field in problem_fields:
            if field in example and example[field]:
                problem = str(example[field])
                break
        
        # Find reasoning steps
        step_fields = ["steps", "reasoning", "intermediate_steps", "process", "work"]
        steps = ""
        for field in step_fields:
            if field in example and example[field]:
                steps_value = example[field]
                if isinstance(steps_value, list):
                    steps = "\n".join([f"Step {i+1}: {step}" for i, step in enumerate(steps_value)])
                elif isinstance(steps_value, str):
                    steps = steps_value
                break
        
        # Find final answer
        answer_fields = ["answer", "final_answer", "solution", "output", "result"]
        answer = ""
        for field in answer_fields:
            if field in example and example[field]:
                answer = str(example[field])
                break
        
        # Combine: problem is prompt, steps + answer is response
        prompt = problem
        if steps and answer:
            response = f"{steps}\n\nFinal Answer: {answer}"
        elif steps:
            response = steps
        else:
            response = answer
        
        return prompt, response
    
    def format_problem_solution(self, example: dict) -> tuple[str, str]:
        """
        Format problem-solution datasets.
        
        Common field names:
        - Problem: problem, question, query, task
        - Solution: solution, answer, code, output, response
        - Explanation: explanation, reasoning, description (optional)
        """
        # Find problem
        problem_fields = ["problem", "question", "query", "task", "prompt", "input"]
        problem = ""
        for field in problem_fields:
            if field in example and example[field]:
                problem = str(example[field])
                break
        
        # Find solution
        solution_fields = ["solution", "answer", "code", "output", "response", "result"]
        solution = ""
        for field in solution_fields:
            if field in example and example[field]:
                solution = str(example[field])
                break
        
        # Find optional explanation
        explanation_fields = ["explanation", "reasoning", "description", "notes"]
        explanation = ""
        for field in explanation_fields:
            if field in example and example[field]:
                explanation = str(example[field])
                break
        
        # Combine problem as prompt, solution (+ explanation) as response
        prompt = problem
        if explanation:
            response = f"{solution}\n\nExplanation: {explanation}"
        else:
            response = solution
        
        return prompt, response
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a single example with input tokens, labels, and loss mask.
        
        Returns:
            Dictionary with:
                - input: Input token IDs (prompt + response)
                - labels: Target token IDs (shifted by 1)
                - loss_mask: Binary mask (1 for response tokens, 0 for prompt)
        """
        example = self.data[idx]
        
        # Format based on instruction format
        if self.instruction_format == "alpaca":
            prompt, response = self.format_alpaca(example)
        elif self.instruction_format == "chatml":
            prompt, response = self.format_chatml(example)
        elif self.instruction_format == "sharegpt":
            prompt, response = self.format_sharegpt(example)
        elif self.instruction_format == "prompt_completion":
            prompt, response = self.format_prompt_completion(example)
        elif self.instruction_format == "text":
            prompt, response = self.format_text(example)
        elif self.instruction_format == "cot":
            prompt, response = self.format_cot(example)
        elif self.instruction_format == "reasoning_steps":
            prompt, response = self.format_reasoning_steps(example)
        elif self.instruction_format == "problem_solution":
            prompt, response = self.format_problem_solution(example)
        elif self.instruction_format == "auto":
            prompt, response = self.format_auto(example)
        else:
            raise ValueError(f"Unknown instruction format: {self.instruction_format}")
        
        # Tokenize prompt and response separately
        prompt_tokens = self.tokenizer.encode(prompt, bos=True, eos=False)
        response_tokens = self.tokenizer.encode(response, bos=False, eos=True)
        
        # EOS token is now added by the tokenizer's encode method above
        
        # Combine and truncate to seq_len + 1 (to account for next-token prediction shift)
        # This matches standard pre-training: we need seq_len+1 tokens to create seq_len input and seq_len labels
        full_tokens = prompt_tokens + response_tokens
        prompt_len = len(prompt_tokens)
        
        if len(full_tokens) > self.seq_len + 1:
            # Truncate from the end (response)
            full_tokens = full_tokens[:self.seq_len + 1]
            prompt_len = min(prompt_len, self.seq_len + 1)
        
        # Pad if necessary to seq_len + 1
        padding_len = (self.seq_len + 1) - len(full_tokens)
        if padding_len > 0:
            full_tokens = full_tokens + [self.tokenizer.pad_id] * padding_len
        
        # Create loss mask: 1 for response tokens, 0 for prompt and padding
        loss_mask = [0] * (self.seq_len + 1)
        for i in range(prompt_len, min(len(full_tokens) - padding_len, self.seq_len + 1)):
            loss_mask[i] = 1
        
        # Convert to tensors - use next-token prediction shift
        # input_ids: tokens 0..seq_len-1 (drop last token)
        # labels: tokens 1..seq_len (drop first token)
        # This gives us seq_len input tokens and seq_len label tokens
        input_ids = torch.tensor(full_tokens[:-1], dtype=torch.long)  # Remove last token for input
        labels = torch.tensor(full_tokens[1:], dtype=torch.long)  # Shift by 1 for labels
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # Align with labels
        
        return {
            "input": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
        }


def build_sft_dataloader(
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    job_config: JobConfig,
    infinite: bool = True,
) -> ParallelAwareDataloader:
    """
    Build a dataloader for supervised fine-tuning with instruction data.
    
    Supports:
    - HuggingFace datasets: Set data.hf_dataset_name or use data_prefix as HF dataset
    - Local JSONL files: Set data_prefix to file path
    - Auto format detection or explicit format setting
    
    Config options:
    - data.data_prefix: Path to JSONL file or HF dataset name
    - data.hf_dataset_name: Explicitly specify HF dataset (optional)
    - data.instruction_format: Format (alpaca, chatml, sharegpt, auto)
    - data.dataset_split: Split to use for HF datasets (default: train)
    - data.text_field: Field name for plain text datasets (optional)
    
    Args:
        dp_world_size: Data parallel world size
        dp_rank: Data parallel rank
        tokenizer: Tokenizer instance
        job_config: Job configuration
        infinite: Whether to loop infinitely over the dataset
    
    Returns:
        ParallelAwareDataloader for SFT
    """
    batch_size = job_config.training.local_batch_size
    seq_len = job_config.training.seq_len
    data_source = job_config.data.data_prefix
    instruction_format = getattr(job_config.data, 'instruction_format', 'auto')
    seed = job_config.data.seed
    
    # HuggingFace dataset specific options
    hf_dataset_name = getattr(job_config.data, 'hf_dataset_name', None)
    hf_dataset_config = getattr(job_config.data, 'hf_dataset_config', None)
    dataset_split = getattr(job_config.data, 'dataset_split', 'train')
    text_field = getattr(job_config.data, 'text_field', None)
    
    logger.info(
        f"Building SFT dataloader: "
        f"source={hf_dataset_name or data_source}, "
        f"format={instruction_format}, "
        f"batch_size={batch_size}, "
        f"seq_len={seq_len}"
    )
    
    # Create dataset
    dataset = InstructionDataset(
        data_source=data_source,
        tokenizer=tokenizer,
        seq_len=seq_len,
        instruction_format=instruction_format,
        seed=seed,
        split=dataset_split,
        hf_dataset_name=hf_dataset_name,
        hf_dataset_config=hf_dataset_config,
        text_field=text_field,
    )
    
    # Create dataloader with proper sharding for distributed training
    from torch.utils.data import DistributedSampler
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=dp_world_size,
        rank=dp_rank,
        shuffle=True,
        seed=seed,
    )
    
    def collate_fn(batch):
        """Collate batch into input_dict and labels."""
        input_ids = torch.stack([item["input"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        loss_masks = torch.stack([item["loss_mask"] for item in batch])
        
        return {
            "input": input_ids,
            "loss_mask": loss_masks,
        }, labels
    
    # Create ParallelAwareDataloader for TorchTitan compatibility
    parallel_dataloader = ParallelAwareDataloader(
        dataset,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=0,  # Can be increased for faster data loading
        pin_memory=True,
    )
    
    logger.info(f"SFT dataloader created with {len(dataset)} examples")
    
    return parallel_dataloader
