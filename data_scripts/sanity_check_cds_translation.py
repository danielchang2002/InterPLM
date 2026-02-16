#!/usr/bin/env python3
"""Sanity-check CDS downloads by translating and comparing to SwissProt protein sequences.

For each CDS:
  1. Look up the correct codon table from the mapping TSV
  2. Translate the CDS nucleotide sequence
  3. Compare to the corresponding SwissProt amino acid sequence
  4. Report match statistics

Usage:
    python sanity_check_cds_translation.py
    python sanity_check_cds_translation.py --sample 5000  # random subset for quick check
"""

import argparse
import random
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


def parse_fasta_dict(fasta_path, key_func):
    """Parse FASTA into dict: key_func(header) -> sequence."""
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


def extract_protein_id_from_cds_header(header):
    """Extract protein_id from NCBI fasta_cds_na header.

    Example: >lcl|... [protein_id=AAT09660.1] ... -> AAT09660.1
    """
    m = re.search(r'\[protein_id=([^\]]+)\]', header)
    return m.group(1) if m else None


def extract_uniprot_ac_from_aa_header(header):
    """Extract UniProt accession from SwissProt FASTA header.

    Example: >sp|Q6GZX4|001R_FRG3G ... -> Q6GZX4
    """
    parts = header.split("|")
    if len(parts) >= 2:
        return parts[1]
    return None


def load_mapping(tsv_path):
    """Load mapping TSV. Returns dict: cds_protein_acc -> {uniprot_ac, transl_table, ...}."""
    mapping = {}
    with open(tsv_path) as f:
        header = f.readline().strip().split("\t")
        for line in f:
            fields = line.strip().split("\t")
            if len(fields) < len(header):
                fields.extend([""] * (len(header) - len(fields)))
            row = dict(zip(header, fields))
            mapping[row["cds_protein_acc"]] = row
    return mapping


def translate_cds(nt_seq, table_id):
    """Translate a CDS nucleotide sequence using the given codon table.

    Returns (translated_aa, issue) where issue is None or a string describing a problem.
    """
    # Trim to multiple of 3
    remainder = len(nt_seq) % 3
    if remainder != 0:
        nt_seq = nt_seq[:len(nt_seq) - remainder]

    if len(nt_seq) == 0:
        return "", "empty_sequence"

    try:
        table = int(table_id) if table_id else 1
    except ValueError:
        table = 1

    try:
        translated = str(Seq(nt_seq).translate(table=table))
    except TranslationError as e:
        return "", f"translation_error: {e}"

    return translated, None


def compare_sequences(translated, reference):
    """Compare translated CDS to reference SwissProt protein.

    Returns (status, detail) where status is one of:
      exact_match, match_no_stop, mismatch_short, mismatch_long, partial_match, no_match
    """
    # Strip trailing stop codon from translation
    trans_clean = translated.rstrip("*")
    ref_clean = reference.rstrip("*")

    if trans_clean == ref_clean:
        return "exact_match", ""

    # Check if match ignoring initial methionine differences (some CDS start with alt start codons)
    if len(trans_clean) == len(ref_clean):
        mismatches = sum(1 for a, b in zip(trans_clean, ref_clean) if a != b)
        if mismatches == 1 and trans_clean[1:] == ref_clean[1:]:
            return "start_codon_diff", f"translated={trans_clean[0]} vs ref={ref_clean[0]}"
        if mismatches <= 3:
            positions = [i for i, (a, b) in enumerate(zip(trans_clean, ref_clean)) if a != b]
            return "near_match", f"{mismatches} mismatches at positions {positions}"

    # Length difference
    len_diff = len(trans_clean) - len(ref_clean)
    if abs(len_diff) > 0:
        # Check if one is a prefix of the other
        min_len = min(len(trans_clean), len(ref_clean))
        if min_len > 0:
            prefix_match = sum(1 for a, b in zip(trans_clean[:min_len], ref_clean[:min_len]) if a == b)
            pct = 100 * prefix_match / min_len
            return "length_diff", f"trans={len(trans_clean)} ref={len(ref_clean)} diff={len_diff} prefix_identity={pct:.1f}%"

    return "no_match", f"trans_len={len(trans_clean)} ref_len={len(ref_clean)}"


def main():
    parser = argparse.ArgumentParser(description="Sanity-check CDS translations against SwissProt")
    parser.add_argument("--cds-fasta", type=str, default=str(CDS_FASTA))
    parser.add_argument("--aa-fasta", type=str, default=str(AA_FASTA))
    parser.add_argument("--mapping-tsv", type=str, default=str(MAPPING_TSV))
    parser.add_argument("--codon-table-tsv", type=str, default=str(CODON_TABLE_TSV),
                        help="Taxonomy-derived codon table mapping (from build_codon_table_mapping.py)")
    parser.add_argument("--sample", type=int, default=0, help="Random sample size (0 = all)")
    parser.add_argument("--verbose", action="store_true", help="Print details for non-matches")
    args = parser.parse_args()

    # Load data
    print("Loading mapping TSV...", flush=True)
    mapping = load_mapping(args.mapping_tsv)
    print(f"  {len(mapping):,} entries", flush=True)

    # Load taxonomy-derived codon tables (overrides the default table=1 from NCBI headers)
    codon_table_path = Path(args.codon_table_tsv)
    codon_tables = {}
    if codon_table_path.exists():
        print("Loading taxonomy-derived codon tables...", flush=True)
        with open(codon_table_path) as f:
            f.readline()  # skip header
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    codon_tables[parts[0]] = parts[1]
        print(f"  {len(codon_tables):,} entries", flush=True)
    else:
        print(f"WARNING: {codon_table_path} not found, using table from mapping TSV (likely all table 1)", flush=True)

    print("Loading SwissProt protein sequences...", flush=True)
    aa_seqs = parse_fasta_dict(args.aa_fasta, extract_uniprot_ac_from_aa_header)
    print(f"  {len(aa_seqs):,} sequences", flush=True)

    print("Loading CDS nucleotide sequences...", flush=True)
    cds_seqs = parse_fasta_dict(args.cds_fasta, extract_protein_id_from_cds_header)
    print(f"  {len(cds_seqs):,} sequences", flush=True)

    # Determine which entries to check
    checkable = [pid for pid in cds_seqs if pid in mapping and mapping[pid]["uniprot_ac"] in aa_seqs]
    print(f"\nCheckable entries (CDS + mapping + SwissProt AA): {len(checkable):,}", flush=True)

    if args.sample > 0 and args.sample < len(checkable):
        random.seed(42)
        checkable = random.sample(checkable, args.sample)
        print(f"Sampled {len(checkable):,} entries for checking", flush=True)

    # Check translations
    counts = {}
    examples = {}
    transl_table_dist = {}

    for pid in checkable:
        row = mapping[pid]
        uniprot_ac = row["uniprot_ac"]
        # Prefer taxonomy-derived codon table, fall back to mapping TSV
        table_id = codon_tables.get(uniprot_ac, row.get("transl_table", "1"))
        transl_table_dist[table_id] = transl_table_dist.get(table_id, 0) + 1

        nt_seq = cds_seqs[pid].upper()
        ref_aa = aa_seqs[uniprot_ac]

        translated, issue = translate_cds(nt_seq, table_id)
        if issue:
            status = issue
            detail = f"nt_len={len(nt_seq)}"
        else:
            status, detail = compare_sequences(translated, ref_aa)

        counts[status] = counts.get(status, 0) + 1

        if status not in ("exact_match", "start_codon_diff") and status not in examples:
            examples[status] = (pid, uniprot_ac, table_id, detail)

        if args.verbose and status not in ("exact_match", "start_codon_diff"):
            print(f"  {status}: {pid} -> {uniprot_ac} (table={table_id}) {detail}", flush=True)

    # Report
    total = len(checkable)
    print(f"\n{'='*60}")
    print(f"TRANSLATION SANITY CHECK RESULTS ({total:,} entries)")
    print(f"{'='*60}")

    for status in sorted(counts, key=counts.get, reverse=True):
        n = counts[status]
        pct = 100 * n / total if total > 0 else 0
        print(f"  {status:25s}: {n:>8,} ({pct:5.1f}%)")

    good = counts.get("exact_match", 0) + counts.get("start_codon_diff", 0) + counts.get("near_match", 0)
    print(f"\n  Overall pass rate: {good:,}/{total:,} ({100*good/total:.1f}%)")

    print(f"\nCodon table distribution:")
    for table_id in sorted(transl_table_dist, key=transl_table_dist.get, reverse=True):
        print(f"  table {table_id:>3s}: {transl_table_dist[table_id]:>8,}")

    if examples:
        print(f"\nExample failures:")
        for status, (pid, uniprot_ac, table_id, detail) in examples.items():
            print(f"  {status}: protein_id={pid}, uniprot={uniprot_ac}, table={table_id}, {detail}")

    # Summary stats
    print(f"\nCoverage:")
    print(f"  CDS downloaded:  {len(cds_seqs):,}")
    print(f"  In mapping:      {sum(1 for p in cds_seqs if p in mapping):,}")
    print(f"  With SwissProt:  {len(checkable):,}")
    not_in_cds = sum(1 for row in mapping.values() if row['cds_protein_acc'] not in cds_seqs)
    print(f"  Mapping entries without CDS: {not_in_cds:,}")


if __name__ == "__main__":
    main()
