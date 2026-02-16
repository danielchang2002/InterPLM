#!/usr/bin/env python3
"""Filter CDS FASTA to only entries whose translation matches the SwissProt protein.

Keeps: exact_match, start_codon_diff, near_match (1-3 AA mismatches, same length)
Removes: length_diff, no_match, translation errors

Usage:
    python filter_cds.py
"""

import re
import sys
from pathlib import Path

from Bio.Seq import Seq
from Bio.Data.CodonTable import TranslationError

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "uniprot"
CDS_FASTA = DATA_DIR / "uniprot_sprot_cds.fasta"
AA_FASTA = DATA_DIR / "uniprot_sprot.fasta"
MAPPING_TSV = DATA_DIR / "uniprot_sprot_cds_mapping.tsv"
CODON_TABLE_TSV = DATA_DIR / "uniprot_codon_tables.tsv"

OUT_FASTA = DATA_DIR / "uniprot_sprot_cds_filtered.fasta"
OUT_MAPPING = DATA_DIR / "uniprot_sprot_cds_mapping_filtered.tsv"


def parse_fasta_with_headers(fasta_path, key_func):
    """Parse FASTA into dict: key -> (full_header, sequence)."""
    seqs = {}
    current_key = None
    header = None
    chunks = []
    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                if current_key is not None:
                    seqs[current_key] = (header, "".join(chunks))
                current_key = key_func(line)
                header = line.rstrip("\n")
                chunks = []
            else:
                chunks.append(line.strip())
    if current_key is not None:
        seqs[current_key] = (header, "".join(chunks))
    return seqs


def parse_fasta_seqs(fasta_path, key_func):
    """Parse FASTA into dict: key -> sequence."""
    seqs = {}
    current_key = None
    chunks = []
    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                if current_key is not None:
                    seqs[current_key] = "".join(chunks)
                current_key = key_func(line)
                chunks = []
            else:
                chunks.append(line.strip())
    if current_key is not None:
        seqs[current_key] = "".join(chunks)
    return seqs


def extract_protein_id(header):
    m = re.search(r'\[protein_id=([^\]]+)\]', header)
    return m.group(1) if m else None


def extract_uniprot_ac(header):
    parts = header.split("|")
    return parts[1] if len(parts) >= 2 else None


def main():
    # Load codon tables
    print("Loading codon tables...", flush=True)
    codon_tables = {}
    with open(CODON_TABLE_TSV) as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                codon_tables[parts[0]] = int(parts[1])
    print(f"  {len(codon_tables):,} entries", flush=True)

    # Load mapping
    print("Loading mapping TSV...", flush=True)
    mapping = {}  # protein_acc -> {uniprot_ac, ...}
    mapping_lines = {}  # protein_acc -> raw TSV line
    with open(MAPPING_TSV) as f:
        tsv_header = f.readline()
        header_fields = tsv_header.strip().split("\t")
        for line in f:
            fields = line.strip().split("\t")
            if len(fields) < len(header_fields):
                fields.extend([""] * (len(header_fields) - len(fields)))
            row = dict(zip(header_fields, fields))
            pid = row["cds_protein_acc"]
            mapping[pid] = row
            mapping_lines[pid] = line
    print(f"  {len(mapping):,} entries", flush=True)

    # Load SwissProt AA sequences
    print("Loading SwissProt protein sequences...", flush=True)
    aa_seqs = parse_fasta_seqs(AA_FASTA, extract_uniprot_ac)
    print(f"  {len(aa_seqs):,} sequences", flush=True)

    # Load CDS with headers
    print("Loading CDS nucleotide sequences...", flush=True)
    cds_data = parse_fasta_with_headers(CDS_FASTA, extract_protein_id)
    print(f"  {len(cds_data):,} sequences", flush=True)

    # Filter
    counts = {"exact_match": 0, "start_codon_diff": 0, "near_match": 0,
              "length_diff": 0, "no_match": 0, "translation_error": 0,
              "missing_ref": 0}
    kept = 0

    with open(OUT_FASTA, "w") as fout, open(OUT_MAPPING, "w") as tout:
        tout.write(tsv_header)

        for pid, (header, nt_seq) in cds_data.items():
            if pid not in mapping:
                continue
            uniprot_ac = mapping[pid]["uniprot_ac"]
            if uniprot_ac not in aa_seqs:
                counts["missing_ref"] += 1
                continue

            table = codon_tables.get(uniprot_ac, 1)
            ref_aa = aa_seqs[uniprot_ac]

            # Translate
            seq = nt_seq.upper()
            remainder = len(seq) % 3
            if remainder:
                seq = seq[:len(seq) - remainder]
            if not seq:
                counts["translation_error"] += 1
                continue

            try:
                translated = str(Seq(seq).translate(table=table))
            except (TranslationError, Exception):
                counts["translation_error"] += 1
                continue

            trans_clean = translated.rstrip("*")
            ref_clean = ref_aa.rstrip("*")

            # Classify
            if trans_clean == ref_clean:
                status = "exact_match"
            elif len(trans_clean) == len(ref_clean):
                mismatches = sum(1 for a, b in zip(trans_clean, ref_clean) if a != b)
                if mismatches == 1 and trans_clean[1:] == ref_clean[1:]:
                    status = "start_codon_diff"
                elif mismatches <= 3:
                    status = "near_match"
                else:
                    status = "no_match"
            else:
                status = "length_diff"

            counts[status] += 1

            # Keep only exact matches and start codon differences
            if status in ("exact_match", "start_codon_diff"):
                fout.write(f"{header}\n")
                # Write sequence in 70-char lines
                for i in range(0, len(nt_seq), 70):
                    fout.write(nt_seq[i:i+70] + "\n")
                tout.write(mapping_lines[pid])
                kept += 1

    total = sum(counts.values())
    print(f"\n{'='*60}", flush=True)
    print(f"FILTER RESULTS ({total:,} entries checked)", flush=True)
    print(f"{'='*60}", flush=True)
    for status, n in sorted(counts.items(), key=lambda x: -x[1]):
        pct = 100 * n / total if total else 0
        marker = " *KEPT*" if status in ("exact_match", "start_codon_diff") else ""
        print(f"  {status:25s}: {n:>8,} ({pct:5.1f}%){marker}")
    print(f"\n  Kept: {kept:,} / {total:,} ({100*kept/total:.1f}%)")
    print(f"\n  Output: {OUT_FASTA}")
    print(f"  Output: {OUT_MAPPING}", flush=True)


if __name__ == "__main__":
    main()
