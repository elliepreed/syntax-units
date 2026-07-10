import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import evaluate

MODEL_NAME = "google/gemma-3-4b-pt"
DATASET = "clams_ru"
OUTDIR = "multilingual/clams/ru"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("device:", device)
if device == "cuda":
    print("gpu:", torch.cuda.get_device_name(0))

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model.to(device)
model.eval()

results = []

for fname in sorted(os.listdir(f"benchmarks/processed/{DATASET}")):
    if not fname.endswith(".csv"):
        continue

    suite = fname[:-4]
    acc = evaluate(model, tokenizer, DATASET, suite, device)
    results.append((suite, acc))
    print(f"{suite:25s} accuracy={acc:.4f}")

os.makedirs(OUTDIR, exist_ok=True)

out_path = f"{OUTDIR}/accuracy_{DATASET}_gemma-3-4b-pt.txt"
with open(out_path, "w") as f:
    for suite, acc in results:
        f.write(f"{acc:.4f} {suite}\n")

print(f"\nSaved to {out_path}")


