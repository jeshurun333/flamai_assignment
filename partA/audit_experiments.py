import unicodedata
import tiktoken

enc = tiktoken.get_encoding("gpt2")

def load_raw(path):
    with open(path, "r", encoding="utf-8") as f:
        return [unicodedata.normalize("NFC", line.strip()) for line in f if line.strip()]

eng_lines = load_raw("starter_kit/starter_kit/corpus_sample/eng_sample.txt")
hin_lines = load_raw("starter_kit/starter_kit/corpus_sample/hin_sample.txt")

print("=== BASELINE (Intern code) ===")
def intern_calc(lines):
    fert, tpc = [], []
    for line in lines:
        l = line.lower()
        toks = enc.encode(l)
        words = l.split(" ")
        chars = len(l)
        fert.append(len(toks) / len(words))
        tpc.append(len(toks) / chars)
    return sum(fert)/len(fert), sum(tpc)/len(tpc)

eng_fert_base, eng_tpc_base = intern_calc(eng_lines)
hin_fert_base, hin_tpc_base = intern_calc(hin_lines)
print(f"Eng base: fert={eng_fert_base:.4f}, tpc={eng_tpc_base:.4f}")
print(f"Hin base: fert={hin_fert_base:.4f}, tpc={hin_tpc_base:.4f}")
print(f"Ratio base: {hin_fert_base / eng_fert_base:.4f}x")

print("\n=== EXPERIMENT 1: Isolate split(' ') vs split() ===")
def calc_split(lines):
    fert, tpc = [], []
    for line in lines:
        l = line.lower()
        toks = enc.encode(l)
        words = l.split()  # FIXED whitespace split
        chars = len(l)
        fert.append(len(toks) / len(words))
        tpc.append(len(toks) / chars)
    return sum(fert)/len(fert), sum(tpc)/len(tpc)

eng_fert_s, eng_tpc_s = calc_split(eng_lines)
hin_fert_s, hin_tpc_s = calc_split(hin_lines)
print(f"Eng split fix: fert={eng_fert_s:.4f} (delta: {eng_fert_s - eng_fert_base:+.4f})")
print(f"Hin split fix: fert={hin_fert_s:.4f} (delta: {hin_fert_s - hin_fert_base:+.4f})")
print(f"Ratio: {hin_fert_s / eng_fert_s:.4f}x (delta ratio: {hin_fert_s / eng_fert_s - hin_fert_base / eng_fert_base:+.4f})")

print("\n=== EXPERIMENT 2: Isolate lower() removal ===")
def calc_case(lines):
    fert, tpc = [], []
    for line in lines:
        l = line  # NO LOWERCASE
        toks = enc.encode(l)
        words = l.split(" ")
        chars = len(l)
        fert.append(len(toks) / len(words))
        tpc.append(len(toks) / chars)
    return sum(fert)/len(fert), sum(tpc)/len(tpc)

eng_fert_c, eng_tpc_c = calc_case(eng_lines)
hin_fert_c, hin_tpc_c = calc_case(hin_lines)
print(f"Eng no-lower: fert={eng_fert_c:.4f}, tpc={eng_tpc_c:.4f} (delta fert: {eng_fert_c - eng_fert_base:+.4f}, delta tpc: {eng_tpc_c - eng_tpc_base:+.4f})")
print(f"Hin no-lower: fert={hin_fert_c:.4f}, tpc={hin_tpc_c:.4f} (delta fert: {hin_fert_c - hin_fert_base:+.4f}, delta tpc: {hin_tpc_c - hin_tpc_base:+.4f})")
print(f"Ratio: {hin_fert_c / eng_fert_c:.4f}x (delta ratio: {hin_fert_c / eng_fert_c - hin_fert_base / eng_fert_base:+.4f})")

print("\n=== EXPERIMENT 3: Isolate Ratio of Sums (Aggregate) vs Mean of Ratios ===")
def calc_aggregate(lines):
    total_toks, total_words, total_chars = 0, 0, 0
    for line in lines:
        l = line.lower()
        toks = enc.encode(l)
        words = l.split(" ")
        total_toks += len(toks)
        total_words += len(words)
        total_chars += len(l)
    return total_toks / total_words, total_toks / total_chars

eng_fert_agg, eng_tpc_agg = calc_aggregate(eng_lines)
hin_fert_agg, hin_tpc_agg = calc_aggregate(hin_lines)
print(f"Eng aggregate: fert={eng_fert_agg:.4f}, tpc={eng_tpc_agg:.4f} (delta fert: {eng_fert_agg - eng_fert_base:+.4f})")
print(f"Hin aggregate: fert={hin_fert_agg:.4f}, tpc={hin_tpc_agg:.4f} (delta fert: {hin_fert_agg - hin_fert_base:+.4f})")
print(f"Ratio: {hin_fert_agg / eng_fert_agg:.4f}x (delta ratio: {hin_fert_agg / eng_fert_agg - hin_fert_base / eng_fert_base:+.4f})")

print("\n=== EXPERIMENT 4: Isolate NFC vs raw (test if normalization is harmless) ===")
def load_without_nfc(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

eng_lines_nonfc = load_without_nfc("starter_kit/starter_kit/corpus_sample/eng_sample.txt")
hin_lines_nonfc = load_without_nfc("starter_kit/starter_kit/corpus_sample/hin_sample.txt")
eng_fert_nonfc, eng_tpc_nonfc = intern_calc(eng_lines_nonfc)
hin_fert_nonfc, hin_tpc_nonfc = intern_calc(hin_lines_nonfc)
print(f"Eng without NFC: fert={eng_fert_nonfc:.4f}, tpc={eng_tpc_nonfc:.4f} (delta: {eng_fert_nonfc - eng_fert_base:+.4f})")
print(f"Hin without NFC: fert={hin_fert_nonfc:.4f}, tpc={hin_tpc_nonfc:.4f} (delta: {hin_fert_nonfc - hin_fert_base:+.4f})")
print(f"Ratio: {hin_fert_nonfc / eng_fert_nonfc:.4f}x (delta: {hin_fert_nonfc / eng_fert_nonfc - hin_fert_base / eng_fert_base:+.4f})")
