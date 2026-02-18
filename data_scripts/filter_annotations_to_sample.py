#!/usr/bin/env python3
"""Filter the full SwissProt annotations TSV to only the 50K sampled proteins.

Usage:
    python filter_annotations_to_sample.py
"""

import gzip
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ANNOTATIONS_TSV = DATA_DIR / "uniprotkb" / "swissprot_annotations.tsv.gz"
MAPPING_TSV = DATA_DIR / "uniprot" / "swissprot_cds_sample_50k_mapping.tsv"
OUTPUT_TSV = DATA_DIR / "uniprotkb" / "swissprot_sample_50k_annotations.tsv.gz"


def main():
    # Load sample accessions
    print("Loading sample accessions...", flush=True)
    acs = set()
    with open(MAPPING_TSV) as f:
        header = f.readline().strip().split("\t")
        ac_col = header.index("uniprot_ac")
        for line in f:
            acs.add(line.strip().split("\t")[ac_col])
    print(f"  {len(acs):,} unique accessions", flush=True)

    # Filter annotations
    print("Filtering annotations...", flush=True)
    kept = 0
    total = 0
    with gzip.open(ANNOTATIONS_TSV, "rt") as fin, gzip.open(OUTPUT_TSV, "wt") as fout:
        tsv_header = fin.readline()
        fout.write(tsv_header)
        fields = tsv_header.strip().split("\t")
        entry_col = fields.index("Entry")

        for line in fin:
            total += 1
            entry = line.split("\t")[entry_col]
            if entry in acs:
                fout.write(line)
                kept += 1

    print(f"  Kept {kept:,} / {total:,} rows")
    print(f"  Output: {OUTPUT_TSV}")


if __name__ == "__main__":
    main()
