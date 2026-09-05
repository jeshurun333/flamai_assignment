# Lab Notebook: Multilingual Tokenizer Audit & Capacity Reconciliation

**Investigator:** AI Engineering Audit Team  
**Dates:** September 5–6, 2026  
**Status:** Complete  

This lab notebook records the chronological progression of hypotheses, experiments, observations, dead ends, and revisions throughout the audit of `REPORT_v0.md`, `fertility.py`, and the serving capacity load test logs.

---

## Log Entry 01: Initial Inspection of `REPORT_v0.md` & Starter Kit

**Timestamp:** 2026-09-05 23:05  
**Observation:**  
We reviewed `starter_kit/` containing:
- `fertility.py`
- `REPORT_v0.md`
- `corpus_sample/` (`eng_sample.txt`, `hin_sample.txt`)
- `bench/` (`model_spec.md`, `bench_log.csv`)

In `REPORT_v0.md`, the previous intern concluded:
1. Hindi fertility is **5.89x worse** than English, recommending budgeting **6x serving cost** for Hindi.
2. The `tok/char` column agrees (1.579 vs 0.226 = 7.0x), claiming this confirms the per-word number.
3. Asserted root cause: *"Hindi simply has more Unicode characters per word, so any tokenizer will struggle. This is a property of the script, not the tokenizer."*
4. In Section 2, claimed that at batch 16, long prompts give higher throughput (1311 tok/s vs 883 tok/s), concluding that *"throughput improves with prompt length"* and recommending linear scaling to batch 48 for ~3200 tok/s.

**Initial Suspicion:**  
Every one of these four claims violates fundamental systems and linguistic principles:
- Running GPT-2 (an English BPE tokenizer) on Hindi is measuring vocabulary coverage, not script efficiency.
- Long prompts cannot magically increase output generation speed on a memory-bandwidth-bound GPU.
- Linear scaling to batch 48 ignores KV-cache memory limits.

---

## Log Entry 02: Auditing `fertility.py` & The Smoke Test Corpus

**Timestamp:** 2026-09-05 23:15  
**Surprise / Dead End #1: The Sample Corpus is Not Even Parallel!**  
We inspected `eng_sample.txt` and `hin_sample.txt`:
- English line 1: *"Bengaluru International Airport handled record traffic in March."*
- Hindi line 1: *"मुझे सुबह की चाय बहुत पसंद है।"* (I like morning tea very much.)
- English line 2: *"The Quarterly Review meeting moved to Thursday."*
- Hindi line 2: *"बेंगलुरु में आज हल्की बारिश हो रही है।"* (It is raining lightly in Bengaluru today.)

The intern claimed these were parallel line-by-line. They are completely mismatched sentences! Comparing averages across unaligned, unrelated sentences with different content is scientifically invalid.

**Dead End #2: The Unicode Normalization False Alarm**  
- *Hypothesis*: Line 49 has `line = unicodedata.normalize("NFC", line)`. We initially suspected this might alter Hindi combining characters or strip matras, distorting character counts.
- *Experiment*: Ran `audit_experiments.py` comparing metrics with and without NFC on the raw text files.
- *Result*: Delta was exactly $\mathbf{0.0000}$ across all metrics and languages.
- *Revision*: NFC is completely harmless and standard best practice. It looks suspicious to a junior auditor, but is totally benign.

**Dead End #3: `random.seed(1337)`**  
- *Hypothesis*: Lines 21 & 25 import `random` and set seed. Does this imply non-deterministic sentence subsampling?
- *Inspection*: Grepped for `random.` in `fertility.py`. It is never used anywhere else in the file. Dead code, completely harmless.

**Confirmed Bug #1: `line.split(" ")` creates phantom empty strings**  
- Line 62 splits strictly on single spaces.
- In `eng_sample.txt` line 7 (`"Please keep the books  in the cupboard."`) and `hin_sample.txt` line 10 (`"किताबें  अलमारी में रखी हैं।"`), double spaces inject empty strings `""` into `words`.
- Word counts were inflated by +1, depressing fertility on those lines by 0.14 and 0.30 respectively.

**Confirmed Bug #2: `line = line.lower()` asymmetrically inflates English tokens**  
- Indic scripts have no uppercase/lowercase distinction. Lowercasing Hindi is a no-op.
- In English, GPT-2 BPE distinguishes capitalized tokens. Lowercasing `"Bengaluru"` fractured the single merge into `['b', 'eng', 'al', 'uru']`, adding +1 token to English.
- This artificially inflated English fertility from 1.229 to 1.265 (+2.8%), compressing the apparent Hindi/English ratio from 6.06x to 5.89x.

**Confirmed Bug #3: Macro-Average vs Aggregate Ratio of Sums**  
- Computing $\frac{1}{N}\sum \frac{T_i}{W_i}$ gives equal weight to short and long sentences, introducing Jensen's inequality bias. Aggregate ratio of sums ($\frac{\sum T_i}{\sum W_i}$) is the correct mathematical definition of corpus fertility.

---

## Log Entry 03: Constructing the Real Multilingual Evaluation Corpus (A1)

**Timestamp:** 2026-09-06 00:00  
**Requirement:** Assemble at least 4 languages including English, Hindi, and two Dravidian languages (e.g. Kannada, Tamil, Telugu, Malayalam).

**Dead End #4: Hugging Face Hub Access Blocked**  
- Attempted to load `facebook/flores` and `ai4bharat/IN22-Gen` via the Hugging Face `datasets` library.
- Both repositories returned `401 Unauthorized: Dataset is gated`. Unauthenticated requests without an HF token were rejected.

**Breakthrough:**  
We located the official, open Meta AI server archive for the NLLB project:
`https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz` (24.4 MB).
We wrote `prepare_corpus.py` to stream-extract exactly the 6 target languages:
- `eng_Latn` (English)
- `hin_Deva` (Hindi)
- `kan_Knda` (Kannada - Dravidian)
- `tam_Taml` (Tamil - Dravidian)
- `tel_Telu` (Telugu - Dravidian)
- `mal_Mlym` (Malayalam - Dravidian)

All 6 files contain exactly 1,012 strictly aligned parallel sentences.

---

## Log Entry 04: Tokenizer Benchmark & The Agglutination Discovery (A3)

**Timestamp:** 2026-09-06 00:15  
We wrote `run_analysis.py` to benchmark 4 tokenizers (`gpt2`, `cl100k_base`, `xlm-roberta-base`, `qwen2.5`) across 5 denominators (`tok/sentence`, `tok/word`, `tok/byte`, `tok/grapheme`, `tok/char`).

**Major Linguistic Surprise: The Agglutination Discrepancy**  
When examining `xlm-roberta-base`:
- English: 30.30 tokens/sentence, 1.40 tokens/word (21,641 words total)
- Kannada: 40.97 tokens/sentence, 2.58 tokens/word (15,909 words total)

Look at the ratios relative to English:
- Kannada `tok / sentence` ratio: **1.35x**
- Kannada `tok / word` ratio: **1.84x**

*Why does `tok/word` show an 84% markup while `tok/sentence` shows only a 35% markup?*  
Because Kannada is agglutinative! It packs grammatical case suffixes and postpositions onto nouns, using 26% fewer words than English to express the exact same meaning. Dividing tokens by whitespace words unfairly penalizes agglutinative languages.

**The Decisive Finding:**  
- On `gpt2`: Hindi is **7.42x** worse per sentence, Tamil is **15.54x** worse, Kannada is **13.58x** worse.
- On `xlm-roberta-base`: Hindi is only **1.25x** (+25%), Tamil is **1.35x** (+35%), Kannada is **1.35x** (+35%), Malayalam is **1.38x** (+38%).
- The intern's claim that high fertility is an inherent property of the script is completely refuted. Switching to an Indic-aware tokenizer collapses the cost expansion from +600% to +25%–35%.

---

## Log Entry 05: Serving Capacity Reconciliation (Part B)

**Timestamp:** 2026-09-06 00:25  
We analyzed `bench/model_spec.md` and `bench/bench_log.csv`.

**Derivation 1: Exact KV Cache Bytes**  
- $L = 28$ layers, $n_{KV} = 8$, $d_{\text{head}} = 128$, `fp16` (2 bytes).
- Per layer: $2 \times 8 \times 128 \times 2 = 4,096$ bytes.
- Total: $28 \times 4,096 = \mathbf{114,688 \text{ bytes/token}}$ (112 KiB/token).

**Derivation 2: Max Concurrent 4,096-Token Sequences**  
- 1 sequence of 4,096 tokens requires $4096 \times 114,688 = 469,762,048 \text{ bytes}$ (448 MiB).
- Usable GPU memory: $24 \text{ GB} \times 0.92 = 22.08 \text{ GB}$.
- Less weights (8.40 GB) and overhead (1.60 GB) = 12.08 GB.
- Max sequences: $\frac{12.08 \times 10^9}{469,762,048} = \mathbf{25.72 \text{ sequences}}$.

**The "Aha!" Moment in `bench_log.csv`:**  
We examined `kv_cache_util` for the long-context sweep:
- Batch 4: util = 0.16 $\to 4 / 0.16 = \mathbf{25.00}$
- Batch 8: util = 0.31 $\to 8 / 0.31 = \mathbf{25.81}$
- Batch 16: util = 0.62 $\to 16 / 0.62 = \mathbf{25.81}$
- Batch 24: util = 0.93 $\to 24 / 0.93 = \mathbf{25.81}$

The empirical serving engine has an exact capacity of **25.8 sequences**, perfectly matching our theoretical derivation of **25.7 sequences**!

**The Preemption Arithmetic:**  
- At batch 32: $32 - 25.8 = 6.2 \to$ Log shows **`preempted_seqs = 7`**!
- At batch 48: $48 - 25.8 = 22.2 \to$ Log shows **`preempted_seqs = 23`**!

The preemptions are not random noise; they are the exact arithmetic overflow of the KV cache!

**Unmasking `reported_tok_s`:**  
The intern claimed batch 16 long prompts hit 1,311 tok/s vs 883 tok/s for short prompts.
We computed honest output generation goodput:
- Short prompt (512 prompt, 256 gen): $\frac{16 \times 256}{13.91} = \mathbf{294.5 \text{ gen tok/s}}$
- Long prompt (3584 prompt, 512 gen): $\frac{16 \times 512}{49.97} = \mathbf{163.9 \text{ gen tok/s}}$

Long prompts generate output tokens **44% slower**, while latency increases 3.6x! The intern was counting prompt prefill tokens ($87.5\%$ of all tokens), mistaking prompt ingestion for generation speed.

---

## Log Entry 06: Strategy Formulation for Conversational Adaptation (Part C)

**Timestamp:** 2026-09-06 00:35  
We evaluated the 3 options under the constraints:
1. 1× A100-80GB for 2 weeks.
2. 1 native-speaker reviewer (Hindi + Kannada only) for 10 h/week.
3. 3-week timeline, zero API budget.

**Key Constraint Deduction:**  
Reviewer throughput is $\sim$24 pairs/hour $\times$ 20 hours = **480 total reviews** before launch.  
Most critically: Tamil, Telugu, Bengali, Marathi have **zero human reviewers**.
- Path (b) (rewriter) is rejected for latency and entity corruption risks.
- Path (a) (SFT) is rejected as a primary path because retrained weights cannot be vetted in 4 out of 6 languages, risking toxic slang, hallucinations, or catastrophic forgetting.
- Path (c) (Prompt Engineering) is selected as the primary path with a Day-4 Kill Criterion (requiring $\ge 65\%$ win-rate). If killed, we pivot to Path (a) LoRA fine-tuning strictly for Hindi and Kannada.
