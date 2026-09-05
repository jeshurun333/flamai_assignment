#!/usr/bin/env python3
"""
prepare_corpus.py -- Downloads and extracts FLORES-200 multilingual parallel corpus
for English, Hindi, and Dravidian languages (Kannada, Tamil, Telugu, Malayalam).
"""

import io
import os
import tarfile
import urllib.request
import unicodedata

FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "corpus")

TARGET_LANGS = {
    "devtest/eng_Latn.devtest": "eng.txt",
    "devtest/hin_Deva.devtest": "hin.txt",
    "devtest/kan_Knda.devtest": "kan.txt",
    "devtest/tam_Taml.devtest": "tam.txt",
    "devtest/tel_Telu.devtest": "tel.txt",
    "devtest/mal_Mlym.devtest": "mal.txt",
}

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Downloading FLORES-200 archive from {FLORES_URL} (~24MB)...")
    req = urllib.request.Request(FLORES_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        data = resp.read()
    print(f"Downloaded {len(data)} bytes. Extracting target splits...")

    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for member in tar.getmembers():
            for key, out_name in TARGET_LANGS.items():
                if member.name.endswith(key):
                    f = tar.extractfile(member)
                    raw_text = f.read().decode("utf-8")
                    lines = [unicodedata.normalize("NFC", l.strip()) for l in raw_text.splitlines() if l.strip()]
                    out_path = os.path.join(OUT_DIR, out_name)
                    with open(out_path, "w", encoding="utf-8") as out_f:
                        out_f.write("\n".join(lines) + "\n")
                    print(f"Extracted {out_name}: {len(lines)} parallel sentences ({os.path.getsize(out_path)} bytes)")

    print(f"All corpora saved successfully to {OUT_DIR}")

if __name__ == "__main__":
    main()
