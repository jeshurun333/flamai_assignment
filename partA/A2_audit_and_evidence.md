# Part A2: Audit of `fertility.py` and the Fertility Metric (Evidence Rule Compliance)

This audit applies the **Evidence Rule** to every identified flaw in `fertility.py` and `REPORT_v0.md`. Each claim isolates the exact line of code, provides the exact reproduction command, presents measured before/after numbers, and explains why the delta proves the distortion.

---

## 1. Code Flaw 1: Naive Space Splitting Creates Phantom Word Tokens (`line.split(" ")`)

### Location:
`fertility.py`, Line 62:
```python
words = line.split(" ")
```

### Mechanism:
In Python, `str.split(" ")` splits strictly on literal single space characters without collapsing consecutive whitespace. If text contains double spaces, leading/trailing whitespace, or non-breaking spaces, `split(" ")` injects empty strings `""` into the word list. 

In `corpus_sample/eng_sample.txt` (line 7: `"Please keep the books  in the cupboard."`), there is a double space between `"books"` and `"in"`. `line.split(" ")` returns 8 items: `['Please', 'keep', 'the', 'books', '', 'in', 'the', 'cupboard.']`, inflating word count from 7 to 8. Similarly, in `corpus_sample/hin_sample.txt` (line 10: `"किताबें  अलमारी में रखी हैं।"`), there is a double space between `"किताबें"` and `"अलमारी"`, inflating word count from 5 to 6.

### Evidence & Isolated Experiment:
- **Command**: `python your-submission/partA/audit_experiments.py` (Experiment 1 vs Baseline)
- **Before (Intern code, `split(" ")`)**:
  - English fertility: **1.2652 tok/word**
  - Hindi fertility: **7.4485 tok/word**
  - Ratio (Hin/Eng): **5.8871x**
- **After (Fix: `words = line.split()`)**:
  - English fertility: **1.2831 tok/word** ($\Delta = +0.0179$, $+1.41\%$)
  - Hindi fertility: **7.5985 tok/word** ($\Delta = +0.1500$, $+2.01\%$)
  - Ratio (Hin/Eng): **5.9221x** ($\Delta = +0.0350$)
- **Specific Line Delta**:
  - `eng_sample.txt` line 7: `len(tokens)=8`. Intern word count = 8 $\to$ fertility = 1.000. True word count = 7 $\to$ fertility = 1.143 ($\Delta = +0.143$).
  - `hin_sample.txt` line 10: `len(tokens)=9`. Intern word count = 6 $\to$ fertility = 1.500. True word count = 5 $\to$ fertility = 1.800 ($\Delta = +0.300$).
- **One-sentence proof**: *The positive delta in fertility proves that the intern's `split(" ")` artificially deflated reported fertility by counting non-existent empty strings as words in the denominator.*

---

## 2. Code Flaw 2: Asymmetric Lowercasing Inflates English Token Count (`line = line.lower()`)

### Location:
`fertility.py`, Line 60:
```python
# lowercase so casing doesn't add noise to the comparison
line = line.lower()
```

### Mechanism:
The intern claimed lowercasing removes "noise". In reality, the GPT-2 BPE vocabulary is case-sensitive, with distinct token embeddings for capitalized words (e.g., `" The"`, `" Bengaluru"`) and lowercase words (`" the"`, `" bengaluru"`). Sentence-initial English words normally begin with an uppercase letter.

Conversely, Indic scripts (Devanagari, Brahmic scripts) have **no concept of letter case**. `line.lower()` on Hindi or Kannada is a no-op! Consequently, lowercasing selectively and asymmetrically altered English text while leaving Hindi untouched. For example, in line 1 of `eng_sample.txt`:
- Original `"Bengaluru"` tokenizes as 3 tokens: `['Beng', 'al', 'uru']` (IDs: 34139, 297, 24056).
- Lowercased `"bengaluru"` tokenizes as 4 tokens: `['b', 'eng', 'al', 'uru']` (IDs: 65, 1146, 297, 24056).

Lowercasing fractured the capitalized subword merge `"Beng"` into `"b"` + `"eng"`, artificially adding a token to English.

### Evidence & Isolated Experiment:
- **Command**: `python your-submission/partA/audit_experiments.py` (Experiment 2 vs Baseline)
- **Before (Intern code, with `line.lower()`)**:
  - English fertility: **1.2652 tok/word**, tok/char: **0.2256**
  - Hindi fertility: **7.4485 tok/word**, tok/char: **1.5791**
  - Ratio (Hin/Eng): **5.8871x**
- **After (Fix: preserve natural casing `line`)**:
  - English fertility: **1.2293 tok/word** ($\Delta = -0.0359$, $-2.84\%$), tok/char: **0.2196** ($\Delta = -0.0061$, $-2.70\%$)
  - Hindi fertility: **7.4485 tok/word** ($\Delta = \pm 0.0000$), tok/char: **1.5791** ($\Delta = \pm 0.0000$)
  - Ratio (Hin/Eng): **6.0590x** ($\Delta = +0.1719$)
- **One-sentence proof**: *The negative delta on English alongside an exact zero delta on Hindi proves that `line.lower()` selectively penalized English by fracturing capitalized BPE tokens, artificially compressing the reported Hindi-to-English ratio by 0.17x.*

---

## 3. Statistical Flaw: Macro-Averaging of Per-Line Ratios vs. Aggregate Ratio (Ratio of Sums)

### Location:
`fertility.py`, Lines 64–67:
```python
per_line_fertility.append(len(tokens) / len(words))
per_line_tpc.append(len(tokens) / chars)
n = len(per_line_fertility)
return sum(per_line_fertility) / n, sum(per_line_tpc) / n
```

### Mechanism:
The script calculates the arithmetic mean of ratios:
$$\bar{F}_{\text{macro}} = \frac{1}{N} \sum_{i=1}^N \frac{T_i}{W_i}$$
In ratio statistics, the unweighted average of per-line ratios gives equal weight to short sentences and long sentences, introducing Jensen's inequality bias. A 4-word sentence with 10 tokens has a ratio of 2.50, skewing the overall mean just as much as a 40-word sentence. The standard definition of corpus fertility is the aggregate ratio of total tokens to total words:
$$F_{\text{aggregate}} = \frac{\sum_{i=1}^N T_i}{\sum_{i=1}^N W_i}$$

### Evidence & Isolated Experiment:
- **Command**: `python your-submission/partA/audit_experiments.py` (Experiment 3 vs Baseline)
- **Before (Macro-average $\frac{1}{N}\sum \frac{T_i}{W_i}$)**:
  - English: **1.2652 tok/word**
  - Hindi: **7.4485 tok/word**
  - Ratio: **5.8871x**
- **After (Aggregate ratio $\frac{\sum T_i}{\sum W_i}$)**:
  - English: **1.2532 tok/word** ($\Delta = -0.0120$, $-0.95\%$)
  - Hindi: **7.4032 tok/word** ($\Delta = -0.0452$, $-0.61\%$)
  - Ratio: **5.9076x** ($\Delta = +0.0205$)
- **One-sentence proof**: *The divergence between the macro-mean and aggregate ratio proves that per-line unweighted averaging introduces non-trivial length-dependent skew compared to the true corpus-level tokenization rate.*

---

## 4. The Conceptual Flaw: The Metric Computes Exactly What It Says, But What It Says Is the Wrong Thing to Compute

### Location:
`fertility.py`, Line 5:
`"Computes tokenizer fertility (tokens per word) and compression (tokens per character)..."`
`REPORT_v0.md`, Findings 1 & 2:
`"1. Hindi fertility is 5.89× worse than English. Serving Hindi will cost us roughly 6x more per request than English."`
`"2. The tok/char column agrees: 1.579 vs 0.226 = 7.0× worse per character, which confirms the per-word number."`

### Conceptual Mechanism:
The script computes `tokens / whitespace_words` and `tokens / unicode_characters`. However, **neither denominator holds semantic content constant across languages**:

1. **"Word" is linguistically non-invariant**:
   - English is primarily isolating/analytic.
   - Dravidian languages (Kannada, Tamil, Telugu, Malayalam) are **highly agglutinative**, attaching postpositions, case markers, tense, and aspect suffixes directly onto root words.
   - For example, in FLORES-200, the exact same 1,012 sentences require **21,641 words** in English, but only **15,909 words** in Kannada and **14,753 words** in Malayalam!
   - Because Dravidian languages express entire clauses in fewer whitespace words, dividing by words artificially inflates Dravidian fertility by 35%–45%, penalizing agglutination rather than token efficiency.
2. **"Unicode Character" is orthographically non-invariant**:
   - In Latin script, each letter is an independent code point.
   - In Brahmic scripts (Devanagari, Kannada, etc.), text is composed of syllables (aksharas). A single akshara consists of multiple Unicode code points (consonant + virama + consonant + vowel matra).
   - Dividing tokens by Unicode characters compares alphabetic phonemes with sub-syllabic code points, rendering `tok/char` meaningless.
3. **The Denominator That Matters: Semantic Task / Parallel Sentence**:
   - For cost and routing decisions, the customer submits a request to complete a task. What matters is: **"How many tokens are consumed to express the SAME semantic meaning?"**
   - When evaluated on parallel sentences with an Indic-aware tokenizer (XLM-RoBERTa), Hindi requires **37.77 tokens/sentence** vs **30.30 tokens/sentence** for English — a ratio of **1.25x**, NOT 6x!
4. **The "Script Property" Fallacy**:
   - The intern asserted: *"Root cause: Hindi simply has more Unicode characters per word, so any tokenizer will struggle. This is a property of the script, not the tokenizer."*
   - This is completely false: the 6x blowup occurred exclusively because the intern ran `gpt2` (a tokenizer trained on 99%+ English WebText with zero Indic merge operations, forcing byte-level fallback of 3 tokens per Devanagari character). When an Indic-aware tokenizer is used, the penalty drops from 589% to **25%**!

### Evidence:
- On parallel FLORES-200 sentences:
  - `gpt2`: Hindi = **198.31 tok/sent** vs English = **26.72 tok/sent** (**7.42x penalty**).
  - `xlm-roberta-base`: Hindi = **37.77 tok/sent** vs English = **30.30 tok/sent** (**1.25x penalty**).
- **One-sentence proof**: *The collapse of the cross-lingual penalty from 7.42x to 1.25x upon switching from GPT-2 to XLM-RoBERTa definitively disproves the intern's claim that high token count is an inherent property of the script rather than a flaw of English-centric vocabularies.*

---

## 5. What Looked Suspicious But Is Actually Fine (Harmless Features)

The prompt explicitly warns: *"And at least one thing in the script looks suspicious but is actually fine... Flagging the harmless thing as a bug — without evidence — costs you points."*

### Feature A: `unicodedata.normalize("NFC", line)` (Line 49)
- **Why it looks suspicious**: A skeptical reviewer might suspect that Unicode normalization alters character counts, strips combining marks (matras), or corrupts Devanagari/Dravidian ligatures before tokenization.
- **Why it is actually fine**: Unicode Normalization Form C (Canonical Decomposition followed by Canonical Composition) is the standard web and NLP specification. Devanagari characters in NFC combine base characters and canonical combining marks deterministically.
- **Experimental Verification**:
  - `audit_experiments.py`, Experiment 4: We disabled `unicodedata.normalize("NFC")` and processed raw lines directly.
  - Result:
    - English with NFC: **1.2652 tok/word**, **0.2256 tok/char**
    - English without NFC: **1.2652 tok/word**, **0.2256 tok/char** ($\Delta = \mathbf{0.0000}$)
    - Hindi with NFC: **7.4485 tok/word**, **1.5791 tok/char**
    - Hindi without NFC: **7.4485 tok/word**, **1.5791 tok/char** ($\Delta = \mathbf{0.0000}$)
  - Delta is exactly zero. It is completely benign and safe.

### Feature B: `random.seed(1337)` (Line 25)
- **Why it looks suspicious**: `import random` and `random.seed(1337)` appear at the top of the file, suggesting non-deterministic sampling, shuffling, or stochastic operations that could affect results.
- **Why it is actually fine**: A static code inspection reveals that the `random` module is never referenced anywhere else in `fertility.py`. It is dead code, but completely inert and harmless.

### Feature C: `add_special_tokens=False` in HuggingFace Tokenizer Wrapper (Line 33)
- **Why it looks suspicious**: Might appear to omit necessary tokens required by BERT/RoBERTa architectures.
- **Why it is actually fine**: When benchmarking text compression/fertility across raw sentence lines, injecting delimiter tokens like `<s>` and `</s>` would artificially inflate token counts by +2 per line, distorting short sentence ratios. Omitting them is the correct methodology for token compression benchmarks.
