# AI Team Intern Assignment — The Audit & R&D Parametric Curve Fitting

**Repository:** AI Team Audit & R&D Assignment Solution  
**GitHub:** [https://github.com/jeshurun333/flamai_assignment](https://github.com/jeshurun333/flamai_assignment)  

---

## 1. R&D Parametric Curve Fitting Solution (`xy_data.csv`)

### Desmos Calculator Link:
👉 **[https://www.desmos.com/calculator/rfj91yrxob](https://www.desmos.com/calculator/rfj91yrxob)**

### Extracted Parameter Values:
- **$\theta = 30^\circ = \frac{\pi}{6} \approx 0.523599$ radians**
- **$M = 0.03$**
- **$X = 55.0$**
- **L1 Loss:** $\approx 3.5 \times 10^{-6}$ (exact analytical fit)

### Parametric Submission Equation (LaTeX format):
```latex
\left(t*\cos(0.5236)-e^{0.03\left|t\right|}\cdot\sin(0.3t)\sin(0.5236)+55,\ 42+t*\sin(0.5236)+e^{0.03\left|t\right|}\cdot\sin(0.3t)\cos(0.5236)\right)
```
*(Parameter range: $6 \le t \le 60$)*

- **Solver Script:** [`solve_curve.py`](solve_curve.py)
- **Detailed Solution Writeup:** [`RD_ASSIGNMENT_SOLUTION.md`](RD_ASSIGNMENT_SOLUTION.md)

---

## 2. AI Team Intern Assignment Deliverables (The Audit)

- **`NOTEBOOK.md`**: Chronological lab notebook detailing hypotheses, empirical experiments, results, dead ends, and revisions.
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
# 1. Run R&D curve parameter solver:
python solve_curve.py

# 2. Verify Part A bug isolation & Evidence Rule numbers:
python partA/audit_experiments.py

# 3. Re-run multilingual tokenizer benchmark across 6 languages:
python partA/run_analysis.py

# 4. Verify Part B KV cache arithmetic and log reconciliation:
python partB/calculations.py
```
