# Part A1: Multilingual Evaluation Corpus Documentation

## 1. Corpus Selection & Overview

To replace the unaligned 10-sentence smoke test in `corpus_sample/`, we assembled a standardized, professionally translated multilingual evaluation corpus using the **FLORES-200** benchmark (from the Meta AI / NLLB initiative). 

We extracted parallel sentences across **6 languages**:
- **English (`eng`)**: `eng_Latn.devtest` (High-resource Indo-European / Germanic baseline)
- **Hindi (`hin`)**: `hin_Deva.devtest` (Primary Indo-Aryan language, Devanagari script)
- **Kannada (`kan`)**: `kan_Knda.devtest` (Dravidian language, Kannada script)
- **Tamil (`tam`)**: `tam_Taml.devtest` (Dravidian language, Tamil script)
- **Telugu (`tel`)**: `tel_Telu.devtest` (Dravidian language, Telugu script)
- **Malayalam (`mal`)**: `mal_Mlym.devtest` (Dravidian language, Malayalam script)

All 6 corpora are strictly sentence-aligned across all 1,012 lines.

---

## 2. Corpus Statistics & Size

| Language | ISO 639-3 / Script | Sentences | Whitespace Words | Unicode Chars | UTF-8 Bytes | Grapheme Clusters | Mean Words / Sent | Mean Bytes / Sent |
|---|---|---|---|---|---|---|---|---|
| **English** | `eng_Latn` | 1,012 | 21,641 | 130,396 | 130,529 | 130,401 | 21.38 | 128.98 |
| **Hindi** | `hin_Deva` | 1,012 | 25,339 | 127,159 | 333,437 | 84,938 | 25.04 | 329.48 |
| **Kannada** | `kan_Knda` | 1,012 | 15,909 | 90,833 | 370,890 | 89,299 | 15.72 | 366.49 |
| **Tamil** | `tam_Taml` | 1,012 | 16,576 | 99,578 | 416,634 | 98,541 | 16.38 | 411.70 |
| **Telugu** | `tel_Telu` | 1,012 | 16,737 | 87,419 | 349,468 | 75,659 | 16.54 | 345.32 |
| **Malayalam**| `mal_Mlym` | 1,012 | 14,753 | 82,319 | 406,859 | 78,486 | 14.58 | 402.03 |

### Key Linguistic Observations:
1. **Agglutination in Dravidian Languages**: Notice that Kannada, Tamil, Telugu, and Malayalam have only **14,700–16,700 whitespace words** for the exact same 1,012 sentences where English requires **21,641 words** and Hindi requires **25,339 words**. Dravidian languages are heavily agglutinative, compounding nouns with case suffixes, postpositions, and verb inflections.
2. **Byte Density of Brahmic Scripts**: In UTF-8, English characters occupy ~1 byte per character (130,529 bytes for 130,396 characters). Indic scripts require 3 bytes per Unicode code point, resulting in ~330KB–417KB per corpus.
3. **Grapheme Clusters vs Code Points**: In Hindi, 127,159 Unicode code points correspond to only 84,938 orthographic syllables (grapheme clusters `\X`). Dividing by raw Unicode code points severely distorts physical text length.

---

## 3. Domain & Data Provenance

- **Source Domain**: Multi-domain web texts curated by professional translators for Wikimedia, news, travel, and health. Sentences were translated by professional native-speaker human linguists with strict quality controls (dual translation, proofreading, and quality checks).
- **Format**: Plain text files with one parallel sentence per line, UTF-8 encoded without BOM.

---

## 4. Preprocessing

The raw texts were preprocessed with minimal non-destructive normalization:
1. **Unicode NFC Normalization**: Every line was passed through `unicodedata.normalize("NFC", line)` to compose canonical combining characters and ensure consistent byte representations across platform filesystems.
2. **Whitespace Stripping**: Trailing whitespace and newline artifacts were stripped using `line.strip()`.
3. **Strict Alignment Check**: Verified that every file contains exactly 1,012 non-empty lines, preserving 1:1 cross-lingual semantic equivalence.
4. **No Destructive Lowercasing**: Unlike the intern's script, we did **not** perform lowercasing, preserving proper capitalization for English acronyms and entities, while avoiding artificial token splits on capitalized subwords.

---

## 5. What This Corpus Cannot Tell You (Caveats as Signal)

While FLORES-200 provides a high-quality, linguistically rigorous parallel baseline for multi-domain news and encyclopedic text, leadership must understand the explicit boundaries of this evaluation:

1. **Domain Mismatch with Production Traffic**: FLORES-200 consists of polished, formal, grammatically pristine sentences translated from international news and Wikipedia. In our production assistant service, user queries and chat responses are informal, colloquial, conversational, and frequently multi-turn. Production queries will feature heavy **code-mixing** (e.g., Hinglish, Tanglish, Kanglish), transliteration in Latin script (e.g., Hindi written in English letters), abbreviations, and emojis. FLORES-200 cannot tell us how tokenizers perform on transliterated text or mixed-script conversational exchanges.
2. **Prompt-to-Completion Ratio Variance**: FLORES-200 measures sentence-level expansion in parallel text. In production LLM serving, user prompts are typically short (10–50 tokens) while system instructions and retrieved context (RAG) can be long (1,000–3,000 tokens), and generated completions vary widely. A tokenizer's cross-lingual cost impact depends on whether Indic tokens dominate the prompt (prefill) or the completion (decode).
3. **Regional Dialects and Slang**: The human translations in FLORES-200 adhere to standardized literary dialects (e.g., Standard Hindi, Formal Tamil). They contain virtually no regional slang (Bambaiya Hindi, Chennai Tamil slang, Bangalore casual Kannada) or conversational discourse markers. Evaluating casualization (Part C) requires dedicated conversational datasets.
