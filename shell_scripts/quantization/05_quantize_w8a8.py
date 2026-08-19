#!/usr/bin/env python3

import os
import time

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier


# ----------------------------------------------------------------------
# Offline mode
# ----------------------------------------------------------------------

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

MODEL_PATH = "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"

SAVE_DIR = (
    "/root/autodl-tmp/qwen25_quant/"
    "quantized_models/Qwen2.5-7B-Instruct-W8A8-smoke"
)

NUM_CALIBRATION_SAMPLES = 32
MAX_SEQUENCE_LENGTH = 512


# ----------------------------------------------------------------------
# Load model
# ----------------------------------------------------------------------

print("========================================")
print("Qwen2.5-7B W8A8 Smoke Quantization")
print(f"Model:               {MODEL_PATH}")
print(f"Save:                {SAVE_DIR}")
print(f"Calibration samples: {NUM_CALIBRATION_SAMPLES}")
print(f"Max sequence length: {MAX_SEQUENCE_LENGTH}")
print("========================================")


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)


model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True,
)


# ----------------------------------------------------------------------
# Build local calibration data
# ----------------------------------------------------------------------

base_prompts = [
    "Explain the difference between CPU and GPU parallel computing.",
    "Describe how KV Cache improves large language model inference.",
    "Explain continuous batching in an inference engine.",
    "Describe the main stages of transformer inference.",
    "Explain the difference between prefill and decode.",
    "What causes GPU memory bandwidth bottlenecks?",
    "Explain tensor parallelism in large language model inference.",
    "Describe how CUDA Graph reduces CPU launch overhead.",
    "Explain INT8 quantization and its advantages.",
    "Describe the role of activation quantization.",
    "Explain why outliers can hurt INT8 quantization accuracy.",
    "Describe the basic idea behind SmoothQuant.",
    "Explain how matrix multiplication is executed on a GPU.",
    "Describe the relationship between latency and throughput.",
    "Explain what TTFT and TPOT measure.",
    "Describe how dynamic batching improves throughput.",
]


messages_list = []

for i in range(NUM_CALIBRATION_SAMPLES):
    prompt = base_prompts[i % len(base_prompts)]

    if i >= len(base_prompts):
        prompt = (
            prompt
            + " Give a detailed technical explanation with an example "
            + f"and focus on scenario {i}."
        )

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    messages_list.append(text)


dataset = Dataset.from_dict(
    {
        "text": messages_list,
    }
)


# ----------------------------------------------------------------------
# Tokenization
# ----------------------------------------------------------------------

def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


dataset = dataset.map(
    tokenize,
    remove_columns=dataset.column_names,
)


print()
print("Calibration dataset ready.")
print(f"Samples: {len(dataset)}")


# ----------------------------------------------------------------------
# W8A8 recipe
# ----------------------------------------------------------------------

recipe = [
    SmoothQuantModifier(
        smoothing_strength=0.8,
    ),

    GPTQModifier(
        targets="Linear",
        scheme="W8A8",
        ignore=["lm_head"],
        offload_hessians=True,
    ),
]


# ----------------------------------------------------------------------
# Quantization
# ----------------------------------------------------------------------

torch.cuda.reset_peak_memory_stats()

start = time.time()

oneshot(
    model=model,
    dataset=dataset,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

elapsed = time.time() - start

peak_memory = torch.cuda.max_memory_allocated() / (1024**3)


print()
print("========================================")
print("Quantization finished")
print(f"Elapsed:         {elapsed:.2f} s")
print(f"Peak GPU memory: {peak_memory:.2f} GB")
print("========================================")


# ----------------------------------------------------------------------
# Save compressed checkpoint
# ----------------------------------------------------------------------

os.makedirs(SAVE_DIR, exist_ok=True)

model.save_pretrained(
    SAVE_DIR,
    save_compressed=True,
)

tokenizer.save_pretrained(SAVE_DIR)


print()
print(f"Saved quantized model to:")
print(SAVE_DIR)