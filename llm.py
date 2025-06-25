import argparse
from dataclasses import dataclass
from collections import defaultdict
import json
import re
import copy
import time
import multiprocessing
from pathlib import Path
import concurrent.futures
from enum import Enum
import asyncio
from typing import cast
import traceback
import math
import random
import datetime
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
from transformers.cache_utils import DynamicCache
from tqdm import tqdm

from common import Logger, CellRule


class SingleModel:
    def __init__(
            self,
            args: argparse.Namespace,
            model,
            tokenizer,
            prompt_path: str = "mutation_prompt.txt",
    ):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.prompt = Path(prompt_path).read_text()

    def generate(self, formulas: list[str]) -> tuple[list[str], list[str]]:
        texts = [self.prompt.format(formula) for formula in formulas]
        assert all(formula in text for formula, text in zip(formulas, texts))
        encoded = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=False
        )
        encoded = {k: v.to(self.model.device) for k, v in encoded.items()}

        with torch.inference_mode():
            outputs = self.model.generate(
                **encoded,
                do_sample=True,
                temperature=self.args.temperature,
                top_k=self.args.top_k,
                top_p=self.args.top_p,
                max_new_tokens=self.args.max_new_tokens,
                min_new_tokens=self.args.min_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        # Extract only the new tokens (exclude prompt tokens).
        output_tokens = outputs[:, encoded["input_ids"].shape[1]:].tolist()
        proof_decoded = []
        for proof_tokens_single in output_tokens:
            proof = self.tokenizer.decode(proof_tokens_single, skip_special_tokens=True)
            # Note: this is a hack because the space is being lost somewhere (is in all_decoded but not in proof_decoded).
            proof = " " + proof
            proof_decoded.append(proof)

        # Log the whole output including prompt and thinking.
        all_decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        del outputs
        del output_tokens
        torch.cuda.empty_cache()

        return proof_decoded, all_decoded


_model: SingleModel | None = None

def _process_init(args: argparse.Namespace, device: str):
    global _model
    if _model is not None:
        print(f"Model already loaded for device {device}!")
        return

    dev_idx = int(device.split(":")[-1])
    print(f"Setting device {dev_idx}...")
    torch.cuda.set_device(dev_idx)

    print(f"Loading model for device {device}")
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, torch_dtype="auto")
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, padding_side="left")
    if tokenizer.pad_token is None:
        print("Setting pad_token = eos_token")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    _model = SingleModel(args, model, tokenizer)
    print(f"Model created for device {device}")

def generate_on_device(sub_formulas: list[str]):
    return _model.generate(sub_formulas)

class ModelProvider:
    def __init__(
            self,
            args: argparse.Namespace,
    ):
        self.args = args
        self.checkpoint = args.checkpoint
        self.per_device_batch_size = args.per_device_batch_size

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")
        n_gpus = torch.cuda.device_count()
        if n_gpus == 0:
            raise RuntimeError("No GPUs available.")
        if args.max_gpus > 0:
            n_gpus = min(n_gpus, args.max_gpus)
        print(f"Will use {n_gpus} GPU(s) for inference.")
        self.n_gpus = n_gpus

        self.executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.n_gpus,
            # CUDA context must not be shared between processes, otherwise we get:
            # "Cannot re-initialize CUDA in forked subprocess"
            mp_context=multiprocessing.get_context("spawn"),
        )

        self._entered = False

    def __enter__(self):
        if self._entered:
            raise RuntimeError("ModelProvider can only be entered once.")
        list(self.executor.map(
            _process_init,
            [self.args] * self.n_gpus,
            [f"cuda:{i}" for i in range(self.n_gpus)]
        ))
        self._entered = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.executor.shutdown(wait=True)

    def generate(self, formulas: list[str]) -> list[str]:
        """
        Splits `prompts` into batches of size `per_device_batch_size`, 
        uses only as many GPUs as needed, runs generation in parallel,
        and returns the concatenated outputs.
        """
        if not self._entered:
            raise RuntimeError("ModelProvider must be entered first.")
        batch_size = self.per_device_batch_size
        if len(formulas) > batch_size * self.n_gpus:
            raise ValueError(
                f"Number of formulas ({len(formulas)}) can be at most "
                f"per_device_batch_size ({batch_size}) × #GPUs ({self.n_gpus})."
            )

        chunks = [
            arr.tolist()
            for arr in np.array_split(formulas, self.n_gpus)
            if len(arr) > 0
        ]

        print(f"Generating {len(chunks)} batches (sizes {', '.join(str(len(c)) for c in chunks)}) ...")
        per_gpu_outputs = list(self.executor.map(generate_on_device, chunks))

        Logger.Instance().log_model_outputs([sample for _, debug_log in per_gpu_outputs for sample in debug_log])
        return [out for batch, _ in per_gpu_outputs for out in batch]

    def mutate(self, rules: list[CellRule]) -> list[list[CellRule]]:
        outputs = self.generate([self._obfuscate(rule.body) for rule in rules])
        results = []
        for rule, output in zip(rules, outputs):
            mutations = self._parse_code_blocks(output)
            results.append([
                CellRule.create(
                    body=self._deobfuscate(mutated),
                    cell_type=rule.cell_type,
                    parent_id=rule.id,
                ) for mutated in mutations
            ])
        return results

    def _obfuscate(self, expr: str) -> str:
        expr = expr.replace("g.get", "g")
        expr = expr.replace("exists_foo c", "exists_foo")
        return expr
    
    def _deobfuscate(self, expr: str) -> str:
        expr = expr.replace("g", "g.get")
        expr = expr.replace("exists_foo", "exists_foo c")
        return expr

    def _parse_code_blocks(self, output: str) -> list[str]:
        blocks = []
        curr_block = None
        for line in output.splitlines():
            if line.startswith("//") or line.startswith("--") or line.startswith("#"):
                # Skip comments.
                continue

            if line.startswith("```lean"):
                curr_block = []
            elif line.startswith("```") and curr_block is not None:
                blocks.append("\n".join(curr_block))
                curr_block = None
            elif curr_block is not None:
                curr_block.append(line)
        return blocks
