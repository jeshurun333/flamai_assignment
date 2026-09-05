# AI Usage Disclosure (`AI_USAGE.md`)

This document provides a candid, transparent breakdown of where AI tools assisted this audit, where AI generated misleading initial hypotheses, and how empirical verification resolved every ambiguity.

---

## 1. Where AI Assisted Effectively

1. **Automating Repetitive Benchmark Scripts**:
   - AI generated the boilerplate for downloading and unpacking the official Meta AI FLORES-200 tarball (`urllib.request` and `tarfile` streaming).
   - AI generated the initial Pandas aggregation pipeline and regex matching (`\X`) for Unicode grapheme cluster counting across 6 languages.
2. **Rapid Mathematical Sizing Verification**:
   - AI helped cross-verify the tensor dimension formulas for Grouped Query Attention (GQA): $2 \times n_{KV} \times d_{\text{head}} \times \text{bytes} \times L = 2 \times 8 \times 128 \times 2 \times 28 = 114,688 \text{ bytes/token}$.
   - AI quickly computed the theoretical KV-cache capacity ($12.08 \text{ GB} / 469.76 \text{ MB} = 25.72 \text{ sequences}$), confirming that our manual derivation was exact.
3. **Drafting Markdown Outlines**:
   - AI helped draft clean, structured Markdown tables and formatted the executive memos according to corporate standards.

---

## 2. Where AI Misled or Generated False Hypotheses

1. **The Unicode Normalization False Flag**:
   - Early in the audit, an AI model pointed to `unicodedata.normalize("NFC", line)` in `fertility.py` and confidently declared: *"NFC normalization alters Devanagari character representations by collapsing independent combining characters, causing an artificial deflation of character counts."*
   - When we ran our isolated experiment (`audit_experiments.py`, Experiment 4), the before/after delta was **exactly 0.0000** across all metrics and languages! Had we accepted the AI's confident assertion without the Evidence Rule, we would have been penalized -5 points for claiming a harmless feature as a bug.
2. **The "Hugging Face Hub Is Sufficient" Trap**:
   - AI initially recommended: *"Just run `datasets.load_dataset('facebook/flores', 'all')` or `ai4bharat/IN22-Gen`."*
   - In practice, both datasets are gated on Hugging Face Hub, causing immediate HTTP 401 errors for unauthenticated execution environments. We had to reject the AI's advice and engineer an independent download pipeline directly from Meta's underlying public file server (`dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz`).
3. **Naive Prompt-Throughput Misinterpretation**:
   - When asked why long prompts had higher reported throughput (1311 vs 883 tok/s), the AI initially hallucinated that *"larger batch tensor dimensions improve GEMM systolic array arithmetic intensity during attention projection."*
   - While arithmetic intensity does increase during prefill, the AI completely missed the operational reality: **`reported_tok_s` was counting input prompt tokens!** The AI treated prompt prefill as output generation. Only through manual inspection of the harness math did we realize that generation goodput had actually plummeted from 295 tok/s down to 164 tok/s.

---

## 3. Key Verification Takeaway

AI tools are high-leverage accelerators for code syntax, mathematical verification, and data formatting. However, when evaluating subtle domain-specific edge cases (Unicode combining marks, scheduler preemption dynamics, and benchmark harness definitions), AI frequently generates convincing, mathematically phrased hallucinations. Every assertion in this submission was subjected to the **Evidence Rule**: isolated, executed via reproducible scripts, and verified against empirical ground-truth numbers.
