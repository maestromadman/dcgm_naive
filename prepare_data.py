"""
prepare_data.py — Build MS MARCO embedding dataset for benchmark_2.py.

Downloads MS MARCO passage corpus from HuggingFace, encodes with
BAAI/bge-base-en-v1.5 (768-dim, L2-normalized), and saves:

    data/embeddings_768.npy   shape (500_000, 768)  float32
    data/queries_768.npy      shape (1_000, 768)     float32

Usage:
    pip install sentence-transformers datasets
    python prepare_data.py
"""

import gc
import os

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import torch

N_CORPUS  = 500_000
N_QUERIES = 1_000
MODEL     = "BAAI/bge-base-en-v1.5"
OUTDIR    = "data"
BATCH     = 512

os.makedirs(OUTDIR, exist_ok=True)

# ── Device ────────────────────────────────────────────────────────────────────
device = "cpu"
try:
    if torch.cuda.is_available():
        torch.zeros(1).cuda()
        device = "cuda"
except Exception:
    pass
print(f"Encoding device: {device}"
      + ("  (GPU unavailable — CPU will be slower)" if device == "cpu" else ""))

# ── Stream MS MARCO ───────────────────────────────────────────────────────────
n_need = N_CORPUS + N_QUERIES
print(f"\nStreaming MS MARCO (need {n_need:,} passages)...")
ds    = load_dataset("Tevatron/msmarco-passage-corpus", split="train", streaming=True)
texts = []
for row in ds:
    text = (row.get("text") or row.get("passage") or "").strip()
    if text:
        texts.append(text)
    if len(texts) % 100_000 == 0 and len(texts):
        print(f"  {len(texts):>7,} / {n_need:,}", flush=True)
    if len(texts) >= n_need:
        break

corpus_texts = texts[:N_CORPUS]
query_texts  = texts[N_CORPUS:]
del texts
gc.collect()

# ── Encode ────────────────────────────────────────────────────────────────────
print(f"\nLoading {MODEL}...")
model = SentenceTransformer(MODEL, device=device)

print(f"Encoding {N_CORPUS:,} corpus passages...")
corpus_emb = model.encode(
    corpus_texts, batch_size=BATCH, show_progress_bar=True,
    convert_to_numpy=True, normalize_embeddings=True,
).astype(np.float32)

print(f"Encoding {N_QUERIES:,} query passages...")
query_emb = model.encode(
    query_texts, batch_size=BATCH, show_progress_bar=True,
    convert_to_numpy=True, normalize_embeddings=True,
).astype(np.float32)

del model, corpus_texts, query_texts
gc.collect()

# ── Save ──────────────────────────────────────────────────────────────────────
out_c = os.path.join(OUTDIR, "embeddings_768.npy")
out_q = os.path.join(OUTDIR, "queries_768.npy")
np.save(out_c, corpus_emb)
np.save(out_q, query_emb)

print(f"\nSaved {out_c}  ({os.path.getsize(out_c) / 1e9:.2f} GB)")
print(f"Saved {out_q}  ({os.path.getsize(out_q) / 1e6:.1f} MB)")
print(f"corpus : {corpus_emb.shape}   queries: {query_emb.shape}")
print("\nDone. Run  python benchmark_2.py  next.")
