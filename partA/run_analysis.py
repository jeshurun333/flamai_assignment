import os
import sys
import unicodedata
import regex
import pandas as pd
import tiktoken
from transformers import AutoTokenizer

# Corpus directory
CORPUS_DIR = r"C:\Users\jeshu\.gemini\antigravity\scratch\your-submission\partA\corpus"

LANGS = {
    "eng": "eng.txt",
    "hin": "hin.txt",
    "kan": "kan.txt",
    "tam": "tam.txt",
    "tel": "tel.txt",
    "mal": "mal.txt",
}

# 1. Load corpora
raw_corpora = {}
for lang, fname in LANGS.items():
    fpath = os.path.join(CORPUS_DIR, fname)
    with open(fpath, "r", encoding="utf-8") as f:
        lines = [unicodedata.normalize("NFC", line.strip()) for line in f if line.strip()]
    raw_corpora[lang] = lines
    print(f"Loaded {lang}: {len(lines)} lines")

# Verify parallel alignment count
num_lines = min(len(lines) for lines in raw_corpora.values())
print(f"Aligning to first {num_lines} parallel sentences across all languages.")
corpora = {lang: lines[:num_lines] for lang, lines in raw_corpora.items()}

# 2. Setup tokenizers
print("\nInitializing tokenizers...")
tokenizers = {}

# GPT-2 (tiktoken)
gpt2_enc = tiktoken.get_encoding("gpt2")
tokenizers["gpt2"] = lambda s: gpt2_enc.encode(s)

# CL100K (tiktoken)
cl100k_enc = tiktoken.get_encoding("cl100k_base")
tokenizers["cl100k_base"] = lambda s: cl100k_enc.encode(s)

# XLM-RoBERTa (SentencePiece unigram)
xlm_tok = AutoTokenizer.from_pretrained("xlm-roberta-base")
tokenizers["xlm-roberta-base"] = lambda s: xlm_tok.encode(s, add_special_tokens=False)

# Qwen 2.5 (BPE 152k vocab)
qwen_tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
tokenizers["qwen2.5"] = lambda s: qwen_tok.encode(s, add_special_tokens=False)

# 3. Compute metrics across all languages and tokenizers
# Grapheme cluster regex pattern: \X matches any extended grapheme cluster
grapheme_pattern = regex.compile(r"\X")

results = []

for tok_name, encode_fn in tokenizers.items():
    print(f"\nProcessing tokenizer: {tok_name}...")
    for lang, lines in corpora.items():
        total_tokens = 0
        total_words = 0
        total_chars = 0
        total_bytes = 0
        total_graphemes = 0
        total_sentences = len(lines)
        
        for line in lines:
            # Token count (without destructive lowercasing)
            toks = encode_fn(line)
            total_tokens += len(toks)
            
            # Words: robust whitespace splitting
            words = line.split()
            total_words += len(words)
            
            # Unicode characters (code points)
            total_chars += len(line)
            
            # UTF-8 bytes
            total_bytes += len(line.encode("utf-8"))
            
            # Grapheme clusters
            total_graphemes += len(grapheme_pattern.findall(line))
            
        # Compute aggregate ratios (Ratio of sums)
        tok_per_word = total_tokens / total_words
        tok_per_char = total_tokens / total_chars
        tok_per_byte = total_tokens / total_bytes
        tok_per_grapheme = total_tokens / total_graphemes
        tok_per_sentence = total_tokens / total_sentences
        
        results.append({
            "tokenizer": tok_name,
            "lang": lang,
            "sentences": total_sentences,
            "tokens": total_tokens,
            "words": total_words,
            "chars": total_chars,
            "bytes": total_bytes,
            "graphemes": total_graphemes,
            "tok_per_word": tok_per_word,
            "tok_per_char": tok_per_char,
            "tok_per_byte": tok_per_byte,
            "tok_per_grapheme": tok_per_grapheme,
            "tok_per_sentence": tok_per_sentence,
        })

df = pd.DataFrame(results)
out_csv = r"C:\Users\jeshu\.gemini\antigravity\scratch\your-submission\partA\corrected_metrics.csv"
df.to_csv(out_csv, index=False)
print(f"\nSaved results to {out_csv}")

# Print pivot tables comparing to English baseline
print("\n" + "="*80)
print("CROSS-LANGUAGE EXPANSION RELATIVE TO ENGLISH (Base = 1.00x)")
print("="*80)

for tok_name in tokenizers:
    sub = df[df["tokenizer"] == tok_name].set_index("lang")
    base_sent = sub.loc["eng", "tok_per_sentence"]
    base_word = sub.loc["eng", "tok_per_word"]
    base_byte = sub.loc["eng", "tok_per_byte"]
    base_char = sub.loc["eng", "tok_per_char"]
    base_graph = sub.loc["eng", "tok_per_grapheme"]
    
    print(f"\n--- Tokenizer: {tok_name} ---")
    summary = pd.DataFrame({
        "tok/sent": sub["tok_per_sentence"],
        "ratio_to_eng (Cost)": sub["tok_per_sentence"] / base_sent,
        "tok/word": sub["tok_per_word"],
        "word_ratio": sub["tok_per_word"] / base_word,
        "tok/byte": sub["tok_per_byte"],
        "tok/grapheme": sub["tok_per_grapheme"],
    })
    print(summary.to_string())
