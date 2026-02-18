#!/usr/bin/env python3
"""Cache Evo2 SAE features for concept evaluation.

Loads pre-computed Evo2 SAE features (nucleotide-level sparse NPZ),
codon mean-pools to amino acid resolution, and saves as shard_N_features.npz
matching the InterPLM annotation shard format.

The Evo2 SAE features are stored as one NPZ per CDS sequence, at nucleotide
resolution. This script:
  1. Reads the annotation shard's aa_metadata.csv to get the ordered protein list
  2. Maps each protein to its NPZ file via the CDS mapping TSV
  3. Loads each NPZ (sparse CSR, float16), codon-pools (mean of 3 nt → 1 aa)
  4. For revcomp: reverses row order after pooling to align with forward AA order
  5. Stacks all proteins and saves as a single shard_N_features.npz
"""

import argparse
import csv
import warnings
from pathlib import Path

import numpy as np
from scipy import sparse
from tqdm import tqdm


def strip_version(acc: str) -> str:
    """Strip version suffix from an accession (e.g., 'FM180568.1' -> 'FM180568')."""
    dot_idx = acc.rfind(".")
    if dot_idx != -1 and acc[dot_idx + 1 :].isdigit():
        return acc[:dot_idx]
    return acc


def parse_npz_filename(path: Path):
    """Extract (embl_acc, cds_protein_acc) from NPZ filename.

    Filename format: lcl|{embl}_cds_{protein}_{idx}_{start}_{length}.npz
    Example: lcl|AAAB01008807.1_cds_EAA04551.4_1_0_1932.npz

    Returns embl_acc WITHOUT version suffix (to match mapping TSV format).
    Returns protein_acc WITH version suffix (already matches mapping TSV).
    """
    stem = path.stem
    parts = stem.split("_cds_")
    if len(parts) != 2:
        return None, None
    embl = parts[0].replace("lcl|", "")
    rest = parts[1]  # e.g. "EAA04551.4_1_0_1932"
    # Split off the trailing _idx_start_length suffix by finding the last 3 underscores
    idx = rest.rfind("_")
    if idx == -1:
        return None, None
    idx2 = rest.rfind("_", 0, idx)
    if idx2 == -1:
        return None, None
    idx3 = rest.rfind("_", 0, idx2)
    if idx3 == -1:
        return None, None
    cds_protein = rest[:idx3]
    # Strip version from EMBL accession to match mapping TSV format
    return strip_version(embl), cds_protein


def load_mapping(mapping_tsv: Path):
    """Load mapping TSV -> dict {uniprot_ac: cds_protein_acc}."""
    mapping = {}
    with open(mapping_tsv) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            ac = row["uniprot_ac"]
            if ac not in mapping:  # take first if duplicates
                mapping[ac] = row["cds_protein_acc"]
    return mapping


def build_feature_index(feature_dir: Path):
    """Scan feature dir -> dict {protein_acc: filepath}.

    Keyed by cds_protein_acc only — EMBL accessions can differ between
    the mapping TSV and the actual NCBI download (301/50k cases observed).
    protein_acc is unique across the 50k sample (verified: zero duplicates).
    """
    index = {}
    npz_files = list(feature_dir.glob("*.npz"))
    for path in tqdm(npz_files, desc="Indexing feature files"):
        _embl, protein = parse_npz_filename(path)
        if protein is not None:
            index[protein] = path
    return index


def load_sparse_npz(path: Path, n_features: int):
    """Load sparse CSR matrix from NPZ, casting float16 to float32."""
    f = np.load(path)
    data = f["data"].astype(np.float32)
    indices = f["indices"]
    indptr = f["indptr"]
    shape = tuple(f["shape"])
    if shape[1] != n_features:
        raise ValueError(
            f"Expected {n_features} features, got {shape[1]} in {path}"
        )
    return sparse.csr_matrix((data, indices, indptr), shape=shape)


def codon_pool(mat_sparse, n_features: int, n_aa: int, has_stop: bool,
               reverse: bool = False):
    """Codon mean-pool nucleotide features to amino acid features.

    Handles three CDS cases:
      - has_stop=True:  CDS has (n_aa+1)*3 nt (stop codon included)
      - has_stop=False: CDS has n_aa*3 nt (no stop codon)
    Trailing nucleotides not forming a complete codon are trimmed before this
    function is called.

    For forward strand with stop: discard the last codon (stop).
    For revcomp with stop: the stop codon RC is at position 0 — discard the
    first codon, pool remaining, then reverse row order.

    Args:
        mat_sparse: CSR matrix (n_nt, n_features), nucleotide-level
        n_features: number of feature columns
        n_aa: expected amino acid count (without stop codon)
        has_stop: whether the CDS includes a stop codon
        reverse: if True, this is revcomp data — reverse after pooling

    Returns:
        CSR matrix (n_aa, n_features)
    """
    dense = mat_sparse.toarray()

    if has_stop:
        if reverse:
            # Revcomp: stop codon RC at the start (first 3 nt)
            dense = dense[3: 3 + n_aa * 3]
        else:
            # Forward: stop codon at the end (last 3 nt)
            dense = dense[: n_aa * 3]
    else:
        # No stop codon — all nucleotides are protein-coding
        dense = dense[: n_aa * 3]

    pooled = dense.reshape(n_aa, 3, n_features).mean(axis=1)  # (n_aa, n_features)
    if reverse:
        pooled = pooled[::-1].copy()
    return sparse.csr_matrix(pooled)


def get_proteins_from_metadata(metadata_path: Path):
    """Read aa_metadata.csv and return ordered list of (uniprot_ac, n_aa) pairs."""
    proteins = []
    current_protein = None
    current_count = 0
    with open(metadata_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            entry = row["Entry"]
            if entry != current_protein:
                if current_protein is not None:
                    proteins.append((current_protein, current_count))
                current_protein = entry
                current_count = 0
            current_count += 1
        if current_protein is not None:
            proteins.append((current_protein, current_count))
    return proteins


def main():
    parser = argparse.ArgumentParser(
        description="Cache Evo2 SAE shard features (codon-pooled to AA resolution)"
    )
    parser.add_argument(
        "--feature_dir",
        type=str,
        required=True,
        help="Directory containing per-CDS NPZ feature files",
    )
    parser.add_argument(
        "--mapping_tsv",
        type=str,
        required=True,
        help="TSV mapping uniprot_ac to embl/cds accessions",
    )
    parser.add_argument(
        "--annot_shard_dir",
        type=str,
        required=True,
        help="Directory containing shard_N/ with aa_metadata.csv",
    )
    parser.add_argument("--shard", type=int, required=True, help="Shard number (0-7)")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for shard_N_features.npz",
    )
    parser.add_argument(
        "--strand",
        type=str,
        required=True,
        choices=["fwd", "revcomp"],
        help="Strand orientation",
    )
    parser.add_argument(
        "--n_features",
        type=int,
        required=True,
        help="Number of SAE features (65536 or 131072)",
    )
    args = parser.parse_args()

    feature_dir = Path(args.feature_dir)
    mapping_tsv = Path(args.mapping_tsv)
    annot_shard_dir = Path(args.annot_shard_dir)
    output_dir = Path(args.output_dir)
    shard = args.shard
    is_revcomp = args.strand == "revcomp"
    n_features = args.n_features

    output_file = output_dir / f"shard_{shard}_features.npz"
    if output_file.exists():
        print(f"Output {output_file} already exists, skipping.", flush=True)
        return

    # 1. Load mapping
    print("Loading mapping TSV...", flush=True)
    mapping = load_mapping(mapping_tsv)
    print(f"  {len(mapping)} proteins in mapping", flush=True)

    # 2. Build feature file index
    print("Building feature file index...", flush=True)
    feat_index = build_feature_index(feature_dir)
    print(f"  {len(feat_index)} feature files indexed", flush=True)

    # 3. Load aa_metadata for this shard -> ordered proteins with AA counts
    metadata_path = annot_shard_dir / f"shard_{shard}" / "aa_metadata.csv"
    print(f"Loading {metadata_path}...", flush=True)
    proteins = get_proteins_from_metadata(metadata_path)
    total_aa = sum(n for _, n in proteins)
    print(f"  {len(proteins)} proteins, {total_aa} total amino acids", flush=True)

    # 4. Process each protein
    protein_matrices = []
    n_found = 0
    n_missing = 0

    for uniprot_ac, n_aa in tqdm(proteins, desc="Codon-pooling proteins"):
        # Look up mapping
        if uniprot_ac not in mapping:
            warnings.warn(
                f"Protein {uniprot_ac} not in mapping TSV, inserting zeros"
            )
            protein_matrices.append(
                sparse.csr_matrix((n_aa, n_features), dtype=np.float32)
            )
            n_missing += 1
            continue

        protein_acc = mapping[uniprot_ac]

        # Find NPZ file by protein accession
        if protein_acc not in feat_index:
            warnings.warn(
                f"No feature file for {uniprot_ac} "
                f"(protein_acc={protein_acc}), inserting zeros"
            )
            protein_matrices.append(
                sparse.csr_matrix((n_aa, n_features), dtype=np.float32)
            )
            n_missing += 1
            continue

        npz_path = feat_index[protein_acc]

        # Load and determine CDS structure
        mat = load_sparse_npz(npz_path, n_features)
        n_nt = mat.shape[0]

        # Trim trailing nucleotides that don't form a complete codon
        remainder = n_nt % 3
        if remainder != 0:
            n_nt_trimmed = n_nt - remainder
            mat = mat[:n_nt_trimmed]
        else:
            n_nt_trimmed = n_nt
        n_codons = n_nt_trimmed // 3

        # Determine if stop codon is present
        if n_codons == n_aa + 1:
            has_stop = True
        elif n_codons == n_aa:
            has_stop = False
        else:
            warnings.warn(
                f"Protein {uniprot_ac}: {n_aa} AA but {n_nt} nt "
                f"({n_codons} codons after trimming) in "
                f"{npz_path.name}, inserting zeros"
            )
            protein_matrices.append(
                sparse.csr_matrix((n_aa, n_features), dtype=np.float32)
            )
            n_missing += 1
            continue

        pooled = codon_pool(
            mat, n_features, n_aa=n_aa, has_stop=has_stop, reverse=is_revcomp
        )
        assert pooled.shape == (n_aa, n_features), (
            f"Shape mismatch: {pooled.shape} != ({n_aa}, {n_features})"
        )
        protein_matrices.append(pooled)
        n_found += 1

    print(f"\nFound: {n_found}, Missing: {n_missing}", flush=True)

    if n_missing > 0:
        print(
            f"WARNING: {n_missing}/{len(proteins)} proteins missing features",
            flush=True,
        )

    # 5. Stack and save
    print("Stacking matrices...", flush=True)
    stacked = sparse.vstack(protein_matrices, format="csr")
    assert stacked.shape == (total_aa, n_features), (
        f"Final shape {stacked.shape} != ({total_aa}, {n_features})"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {output_file}...", flush=True)
    sparse.save_npz(output_file, stacked)
    print(f"Done! Shape: {stacked.shape}, NNZ: {stacked.nnz}", flush=True)


if __name__ == "__main__":
    main()
