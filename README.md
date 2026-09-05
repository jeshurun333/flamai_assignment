# AI Team Intern Assignment — The Audit

**Candidate Submission**  
**Repository:** Multilingual Tokenizer Audit, Capacity Reconciliation & Conversational Tone Adaptation  

---

## Deliverables Index

- **`NOTEBOOK.md`**: Chronological lab notebook detailing the complete inquiry trajectory: hypotheses, empirical experiments, results, dead ends, and revisions.
- **`AI_USAGE.md`**: Transparent disclosure of where AI assisted effectively and where it generated misleading hypotheses.
- **`partA/`**:
  - `prepare_corpus.py`: Downloader and extractor for the 1,012-sentence parallel FLORES-200 evaluation corpus.
  - `corpus/`: Parallel sentences for English, Hindi, Kannada, Tamil, Telugu, and Malayalam.
  - `audit_experiments.py`: Script isolating each bug in `fertility.py` under the Evidence Rule.
  - `run_analysis.py`: Multi-tokenizer (`gpt2`, `cl100k_base`, `xlm-roberta-base`, `qwen2.5`) and multi-denominator benchmark script.
  - `corrected_metrics.csv`: Data table of all measured metrics.
  - `A1_corpus_documentation.md`: Corpus size, linguistic properties, and domain caveats.
  - `A2_audit_and_evidence.md`: Line-by-line script and metric audit with verified deltas and harmless features.
  - `A3_corrected_analysis.md`: Corrected cross-language comparison and denominator reasoning.
  - `memo.md`: Executive recommendation memo for leadership.
- **`partB/`**:
  - `calculations.py`: Script executing exact KV-cache arithmetic and log validations.
  - `capacity_reconciliation.md`: Comprehensive reconciliation of serving setup (B1, B2, B3, B4).
- **`partC/`**:
  - `memo.md`: Strategic decision memo detailing assumptions, arithmetic, success threshold, kill criterion, and day-1 experiment.

---

## Quickstart & Verification

```bash
# 1. Verify Part A bug isolation & Evidence Rule numbers:
python partA/audit_experiments.py

# 2. Re-run multilingual tokenizer benchmark across 6 languages:
python partA/run_analysis.py

# 3. Verify Part B KV cache arithmetic and log reconciliation:
python partB/calculations.py
```
