#!/usr/bin/env python3
"""
calculations.py -- Exact mathematical derivations and log checks for Part B.
"""

import pandas as pd
import numpy as np

# Model Specifications
PARAMS = 4.2e9
LAYERS = 28
D_MODEL = 3072
Q_HEADS = 24
KV_HEADS = 8  # Grouped Query Attention (GQA)
HEAD_DIM = 128
WEIGHTS_PRECISION_BYTES = 2  # fp16
KV_PRECISION_BYTES = 2       # fp16

# Hardware & Runtime Specifications
GPU_TOTAL_MEM_GB = 24.0  # NVIDIA L4 (24 GB)
PEAK_BANDWIDTH_GB_S = 300.0
PEAK_FP16_TFLOPS = 121.0
MAX_MODEL_LEN = 4096
GPU_MEM_UTIL = 0.92
RUNTIME_OVERHEAD_GB = 1.6

print("="*70)
print("B1: KV CACHE ARITHMETIC")
print("="*70)

# (a) KV cache bytes per token exactly:
# For each token, each layer stores Key and Value tensors.
# Key tensor per token per layer: KV_HEADS * HEAD_DIM * KV_PRECISION_BYTES
# Value tensor per token per layer: KV_HEADS * HEAD_DIM * KV_PRECISION_BYTES
# Bytes per token per layer = 2 * KV_HEADS * HEAD_DIM * KV_PRECISION_BYTES
kv_bytes_per_layer = 2 * KV_HEADS * HEAD_DIM * KV_PRECISION_BYTES
kv_bytes_per_token = LAYERS * kv_bytes_per_layer

print(f"Key/Value elements per layer per token: 2 * {KV_HEADS} * {HEAD_DIM} = {2 * KV_HEADS * HEAD_DIM}")
print(f"Bytes per token per layer: {kv_bytes_per_layer} bytes ({kv_bytes_per_layer / 1024:.1f} KiB)")
print(f"Total KV cache bytes per token across {LAYERS} layers:")
print(f"  = {LAYERS} * {kv_bytes_per_layer} = {kv_bytes_per_token:,} bytes")
print(f"  = {kv_bytes_per_token / 1024:.2f} KiB/token")
print(f"  = {kv_bytes_per_token / (1024**2):.6f} MiB/token")

# (b) Approximate maximum number of concurrent 4096-token sequences:
# Model weights size: 4.2B params * 2 bytes = 8.4 GB
weights_gb = (PARAMS * WEIGHTS_PRECISION_BYTES) / 1e9  # 8.4 GB (or / 1024^3 = 7.82 GiB)
weights_bytes = PARAMS * WEIGHTS_PRECISION_BYTES       # 8,400,000,000 bytes

# Available memory for KV cache:
# Usable memory = 24 GB * 0.92 = 22.08 GB
# KV memory = Usable memory - Weights - Runtime Overhead
# Decimal convention (GB = 1e9 bytes):
usable_mem_dec = GPU_TOTAL_MEM_GB * 1e9 * GPU_MEM_UTIL
kv_mem_dec = usable_mem_dec - weights_bytes - (RUNTIME_OVERHEAD_GB * 1e9)

# Binary convention (GiB = 1024^3 bytes, standard in vLLM / PyTorch cudaMemGetInfo):
total_mem_bin = GPU_TOTAL_MEM_GB * (1024**3)
usable_mem_bin = total_mem_bin * GPU_MEM_UTIL
kv_mem_bin = usable_mem_bin - weights_bytes - (RUNTIME_OVERHEAD_GB * (1024**3))

# KV bytes for one 4096-token sequence:
kv_bytes_per_4096 = 4096 * kv_bytes_per_token

max_seqs_dec = kv_mem_dec / kv_bytes_per_4096
max_seqs_bin = kv_mem_bin / kv_bytes_per_4096

print(f"\nKV cache size per 4,096-token sequence:")
print(f"  = 4096 * {kv_bytes_per_token:,} bytes = {kv_bytes_per_4096:,} bytes")
print(f"  = {kv_bytes_per_4096 / (1024**2):.2f} MiB ({kv_bytes_per_4096 / (1024**3):.4f} GiB)")

print(f"\nMemory Allocation (Decimal Basis):")
print(f"  Total Usable (0.92 * 24GB): {usable_mem_dec / 1e9:.2f} GB")
print(f"  Model Weights (4.2B fp16):  {weights_gb:.2f} GB")
print(f"  Runtime Overhead:           {RUNTIME_OVERHEAD_GB:.2f} GB")
print(f"  Available for KV cache:     {kv_mem_dec / 1e9:.2f} GB ({kv_mem_dec:,.0f} bytes)")
print(f"  Max Concurrent 4096 Seqs:   {kv_mem_dec / kv_bytes_per_4096:.2f} => ~25 to 26 sequences")

print(f"\nMemory Allocation (Binary GiB Basis):")
print(f"  Available for KV cache:     {kv_mem_bin / (1024**3):.2f} GiB ({kv_mem_bin:,.0f} bytes)")
print(f"  Max Concurrent 4096 Seqs:   {kv_mem_bin / kv_bytes_per_4096:.2f} => ~27 sequences")

# Verification against log:
log_path = "starter_kit/starter_kit/bench/bench_log.csv"
df = pd.read_csv(log_path)
print("\nLog Verification (Long Prompt 3584+512 = 4096 tokens):")
long_df = df[df["prompt_len"] == 3584].copy()
for idx, r in long_df.iterrows():
    b = int(r["batch_size"])
    u = float(r["kv_cache_util"])
    p = int(r["preempted_seqs"])
    implied_capacity = b / u if u > 0 else np.nan
    print(f"  Batch {b:2d}: kv_cache_util = {u:.2f} | Implied Max Seqs = {implied_capacity:.2f} | Preempted = {p}")

print("\n" + "="*70)
print("B2 & B3: THROUGHPUT, GOODPUT & PREEMPTION MECHANISM")
print("="*70)

for idx, r in long_df.iterrows():
    b = int(r["batch_size"])
    w = float(r["wall_clock_s"])
    rep = float(r["reported_tok_s"])
    p = int(r["preempted_seqs"])
    itl = float(r["itl_ms_p50"])
    ttft = float(r["ttft_ms_p50"])
    
    # Derivation 1: Total Generated Tokens / Wall Clock Time
    goodput_1 = (b * 512) / w
    # Derivation 2: Scaled reported throughput (gen fraction: 512 / 4096 = 1/8)
    goodput_2 = rep * (512 / 4096)
    # Steady-state decode generation rate from ITL:
    decode_rate = b / (itl / 1000.0)
    
    print(f"Batch {b:2d}: wall={w:6.2f}s, reported={rep:7.1f} tok/s | Honest Goodput = {goodput_1:5.1f} tok/s (formula 2: {goodput_2:5.1f}) | ITL decode rate={decode_rate:5.1f} tok/s | preempted={p}")

print("\nShort Prompt Benchmark (512+256 = 768 tokens):")
short_df = df[df["prompt_len"] == 512].copy()
for idx, r in short_df.iterrows():
    b = int(r["batch_size"])
    w = float(r["wall_clock_s"])
    rep = float(r["reported_tok_s"])
    goodput = (b * 256) / w
    print(f"Batch {b:2d}: wall={w:6.2f}s, reported={rep:7.1f} tok/s | Honest Goodput = {goodput:5.1f} tok/s")
