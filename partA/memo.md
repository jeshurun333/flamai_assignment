# Executive Recommendation Memo: Tokenizer Audit & Routing Strategy

**To:** Leadership Team  
**From:** AI Engineering Audit Team  
**Date:** September 6, 2026  
**Subject:** Corrected Tokenizer Economics & Indic Routing Recommendation (Audit of REPORT_v0)

---

### 1. Corrected Headline Numbers

The findings in `REPORT_v0.md` claiming a **5.89x–7.0x serving cost penalty** for Hindi were invalid artifacts of running an English-only tokenizer (`gpt2`) on 10 unaligned sentences with flawed line-by-line metrics.

When evaluated across **1,012 parallel sentences (FLORES-200)** comparing English against Hindi and major Dravidian languages (Kannada, Tamil, Telugu, Malayalam), the true semantic token expansion factors are:

| Language Family | Language | `gpt2` (v0 deck) | `cl100k` (GPT-4) | `xlm-roberta` (Multilingual) | **Corrected Serving Cost Expansion** |
|---|---|---|---|---|---|
| **Baseline** | **English** | 1.00x | 1.00x | 1.00x | **1.00x (Baseline)** |
| **Indo-Aryan** | **Hindi** | 7.42x | 4.77x | **1.25x** | **+25% cost overhead** (not +500%) |
| **Dravidian** | **Kannada** | 13.58x | 8.86x | **1.35x** | **+35% cost overhead** (not +1200%) |
| **Dravidian** | **Tamil** | 15.54x | 7.64x | **1.35x** | **+35% cost overhead** (not +1400%) |
| **Dravidian** | **Telugu** | 12.97x | 8.29x | **1.32x** | **+32% cost overhead** (not +1100%) |
| **Dravidian** | **Malayalam** | 15.16x | 8.94x | **1.38x** | **+38% cost overhead** (not +1400%) |

**Core Takeaway:** With an Indic-aware or modern multilingual tokenizer, the true cost markup for Indic traffic is only **25% to 38%** above English. Budgeting 6x capacity for Hindi is a severe multi-million-dollar over-provisioning error.

---

### 2. Routing & Architecture Recommendation

1. **Reject Separate Indic Model Routing**: Do **not** deploy a separate, fragmented Indic-only model or siloed infrastructure. Operating dual model stacks doubles cold-start overhead, fragments GPU pools, and duplicates operational maintenance.
2. **Adopt a Unified Multilingual Tokenizer**: Upgrade the serving stack to a unified model utilizing an Indic-competent tokenizer (e.g., SentencePiece unigram or modern 128k–152k BPE vocabularies such as Llama-3, Gemma-2, or Qwen-2.5). This immediately achieves single-stack routing with minimal (~1.3x) token expansion.
3. **Budget Capacity at 1.35x for Blended Indic Requests**: Update corporate financial models to provision **1.30x–1.35x token budget per Indic request**, completely retiring the 6x assumption.

---

### 3. The Biggest Caveat

**Script Code-Mixing and Colloquial Transliteration (Hinglish / Romanized Indic):**  
FLORES-200 consists of formal literary text written strictly in native Brahmic scripts. In actual production, Indian consumer traffic exhibits heavy **code-mixing** (alternating between English and Indic words) and **transliteration** (writing Hindi/Kannada phonetically using Latin letters, e.g., *"kya haal hai"*). Multilingual subword tokenizers trained primarily on native scripts can experience unpredictable subword fragmentation on non-standardized Latin transliterations, creating localized token spikes not captured in standard literary benchmarks.

---

### 4. The Single Production Metric to Monitor

**`tokens_per_request_by_language` (specifically: p50 and p95 token consumption per completed interaction tagged by detected query language).**

*How it catches errors:* If our production telemetry shows the ratio $\frac{\text{p50\_tokens}(\text{Indic})}{\text{p50\_tokens}(\text{English})}$ creeping above **1.50x**, it immediately alerts engineering to subword degradation (caused by dialectal slang, code-mixing, or Latin transliteration), triggering targeted vocabulary expansion before capacity limits are breached.
