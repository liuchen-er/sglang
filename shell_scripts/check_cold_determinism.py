import os
import random
import requests
import time

BASE_URL = "http://127.0.0.1:30000"
MODEL_PATH = os.environ["MODEL_PATH"]
PROMPT_LEN = 2048
SEED = 102048
TRIALS = 10
from transformers import AutoConfig

config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
VOCAB_SIZE = config.vocab_size


def make_random_ids(length, seed):
    rng = random.Random(seed)
    return [rng.randrange(1000, min(VOCAB_SIZE - 1, 50000)) for _ in range(length)]


def flush_cache():
    r = requests.post(f"{BASE_URL}/flush_cache", timeout=30)
    r.raise_for_status()
    time.sleep(0.5)


def generate(ids):
    payload = {
        "input_ids": ids,
        "sampling_params": {
            "temperature": 0,
            "top_k": 1,
            "top_p": 1.0,
            "min_p": 0.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "repetition_penalty": 1.0,
            "max_new_tokens": 64,
        },
        "stream": False,
    }
    r = requests.post(f"{BASE_URL}/generate", json=payload, timeout=300)
    r.raise_for_status()
    body = r.json()
    if isinstance(body, list):
        body = body[0]
    return body["text"]


ids = make_random_ids(PROMPT_LEN, SEED)
flush_cache()
cold = generate(ids)
print("Cold:", repr(cold))
outputs = []
for i in range(10):
    output = generate(ids)
    outputs.append(output)
    print(f"L1[{i}]:", repr(output))
print("Unique L1 outputs:", len(set(outputs)))
reference = outputs[0]
same = sum(x == reference for x in outputs)
print()
print(f"Exact same = {same}/{TRIALS}")
print(f"Unique outputs = {len(set(outputs))}")
