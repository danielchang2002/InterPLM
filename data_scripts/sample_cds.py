#!/usr/bin/env python3
"""Sample 50K CDS sequences with protein length < 1024 AA, and generate reverse complements.

Outputs:
  - swissprot_cds_sample_50k.fasta          (sampled CDS, forward)
  - swissprot_cds_sample_50k_revcomp.fasta   (reverse complement)
  - swissprot_cds_sample_50k_mapping.tsv     (mapping for sampled entries)

Usage:
    python sample_cds.py
"""

import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "uniprot"
CDS_FASTA = DATA_DIR / "uniprot_sprot_cds_filtered.fasta"
MAPPING_TSV = DATA_DIR / "uniprot_sprot_cds_mapping_filtered.tsv"

OUT_FWD = DATA_DIR / "swissprot_cds_sample_50k.fasta"
OUT_RC = DATA_DIR / "swissprot_cds_sample_50k_revcomp.fasta"
OUT_MAP = DATA_DIR / "swissprot_cds_sample_50k_mapping.tsv"

SAMPLE_SIZE = 50_000
MAX_AA_LEN = 1024
SEED = 42

COMP = str.maketrans("ACGTacgtNnRYSWKMBDHV", "TGCAtgcaNnYRSWMKVHDB")


def revcomp(seq):
    return seq.translate(COMP)[::-1]


def main():
    # Pass 1: find eligible sequences (protein len < 1024 AA → nt len < 3072)
    max_nt = MAX_AA_LEN * 3
    print(f"Scanning for CDS with protein length < {MAX_AA_LEN} AA (nt < {max_nt})...", flush=True)

    eligible = []  # (header, sequence)
    header = None
    chunks = []

    with open(CDS_FASTA) as f:
        for line in f:
            if line.startswith(">"):
                if header is not None:
                    seq = "".join(chunks)
                    if len(seq) <= max_nt:
                        eligible.append((header, seq))
                header = line.rstrip("\n")
                chunks = []
            else:
                chunks.append(line.strip())
        if header is not None:
            seq = "".join(chunks)
            if len(seq) <= max_nt:
                eligible.append((header, seq))

    print(f"  Eligible: {len(eligible):,} / total sequences", flush=True)

    # Sample
    random.seed(SEED)
    if len(eligible) <= SAMPLE_SIZE:
        print(f"  WARNING: fewer eligible than requested, using all {len(eligible):,}", flush=True)
        sampled = eligible
    else:
        sampled = random.sample(eligible, SAMPLE_SIZE)
    print(f"  Sampled: {len(sampled):,}", flush=True)

    # Build set of sampled protein_ids for mapping filter
    import re
    pid_pattern = re.compile(r'\[protein_id=([^\]]+)\]')
    sampled_pids = set()
    for hdr, _ in sampled:
        m = pid_pattern.search(hdr)
        if m:
            sampled_pids.add(m.group(1))

    # Write forward FASTA
    print(f"Writing {OUT_FWD}...", flush=True)
    with open(OUT_FWD, "w") as f:
        for header, seq in sampled:
            f.write(f"{header}\n")
            for i in range(0, len(seq), 70):
                f.write(seq[i:i+70] + "\n")

    # Write reverse complement FASTA
    print(f"Writing {OUT_RC}...", flush=True)
    with open(OUT_RC, "w") as f:
        for header, seq in sampled:
            rc = revcomp(seq)
            f.write(f"{header} [revcomp=true]\n")
            for i in range(0, len(rc), 70):
                f.write(rc[i:i+70] + "\n")

    # Write filtered mapping
    print(f"Writing {OUT_MAP}...", flush=True)
    kept = 0
    with open(MAPPING_TSV) as fin, open(OUT_MAP, "w") as fout:
        tsv_header = fin.readline()
        fout.write(tsv_header)
        header_fields = tsv_header.strip().split("\t")
        pid_col = header_fields.index("cds_protein_acc")
        for line in fin:
            fields = line.strip().split("\t")
            if len(fields) > pid_col and fields[pid_col] in sampled_pids:
                fout.write(line)
                kept += 1
    print(f"  Mapping entries written: {kept:,}", flush=True)

    # Summary stats
    import numpy as np
    lengths = np.array([len(seq) for _, seq in sampled])
    print(f"\nSample statistics:")
    print(f"  Sequences:    {len(sampled):,}")
    print(f"  Total nt:     {lengths.sum():,} ({lengths.sum()/1e6:.1f} Mb)")
    print(f"  Length (nt):   min={lengths.min()}, median={int(np.median(lengths))}, "
          f"mean={lengths.mean():.0f}, max={lengths.max()}")
    print(f"  Length (AA):   min={lengths.min()//3}, median={int(np.median(lengths))//3}, "
          f"mean={lengths.mean()/3:.0f}, max={lengths.max()//3}")


if __name__ == "__main__":
    main()
