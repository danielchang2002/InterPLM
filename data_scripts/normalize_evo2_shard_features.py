#!/usr/bin/env python3
"""Normalize cached Evo2 SAE shard features by per-feature global max.

Two-pass normalization:
  Pass 1: Compute per-feature max across ALL shards (0-7), excluding the
           first 10 AA (fwd) or last 10 AA (revcomp) of each protein to
           avoid anomalously large activations at sequence boundaries.
  Pass 2: Divide each shard's features by the global max and overwrite.

Run after eval_evo2_cache.sh completes, before eval_evo2_compare.sh.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy import sparse


SKIP_AA = 10  # Number of edge amino acids to exclude from max calculation


def get_protein_lengths(metadata_path: Path):
    """Read aa_metadata.csv and return ordered list of protein AA lengths."""
    lengths = []
    current_protein = None
    current_count = 0
    with open(metadata_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            entry = row["Entry"]
            if entry != current_protein:
                if current_protein is not None:
                    lengths.append(current_count)
                current_protein = entry
                current_count = 0
            current_count += 1
        if current_protein is not None:
            lengths.append(current_count)
    return lengths


def build_keep_mask(protein_lengths, is_fwd):
    """Build boolean mask: True for rows to KEEP in max calculation.

    For fwd:     exclude first SKIP_AA amino acids of each protein.
    For revcomp: exclude last  SKIP_AA amino acids of each protein.
    """
    total_aa = sum(protein_lengths)
    keep = np.ones(total_aa, dtype=bool)
    pos = 0
    for length in protein_lengths:
        skip = min(SKIP_AA, length)  # don't skip more than protein length
        if is_fwd:
            keep[pos : pos + skip] = False
        else:
            keep[pos + length - skip : pos + length] = False
        pos += length
    return keep


def main():
    parser = argparse.ArgumentParser(
        description="Normalize Evo2 SAE shard features by per-feature global max"
    )
    parser.add_argument(
        "--features_dir",
        type=str,
        required=True,
        help="Directory containing shard_N_features.npz (will be overwritten)",
    )
    parser.add_argument(
        "--annot_shard_dir",
        type=str,
        required=True,
        help="Directory containing shard_N/ with aa_metadata.csv",
    )
    parser.add_argument(
        "--strand",
        type=str,
        required=True,
        choices=["fwd", "revcomp"],
        help="Strand orientation (determines which edge AAs to exclude)",
    )
    parser.add_argument(
        "--n_shards",
        type=int,
        default=8,
        help="Number of shards (default: 8)",
    )
    args = parser.parse_args()

    features_dir = Path(args.features_dir)
    annot_shard_dir = Path(args.annot_shard_dir)
    is_fwd = args.strand == "fwd"
    n_shards = args.n_shards

    # --- Pass 1: compute global per-feature max ---
    print("=== Pass 1: Computing per-feature max across all shards ===", flush=True)
    global_max = None

    for shard in range(n_shards):
        feat_path = features_dir / f"shard_{shard}_features.npz"
        if not feat_path.exists():
            print(f"  Shard {shard}: not found, skipping", flush=True)
            continue

        print(f"  Shard {shard}: loading...", flush=True)
        mat = sparse.load_npz(feat_path).tocsr()
        n_features = mat.shape[1]

        if global_max is None:
            global_max = np.zeros(n_features, dtype=np.float64)

        # Get protein boundaries for this shard
        metadata_path = annot_shard_dir / f"shard_{shard}" / "aa_metadata.csv"
        protein_lengths = get_protein_lengths(metadata_path)
        assert sum(protein_lengths) == mat.shape[0], (
            f"Shard {shard}: metadata has {sum(protein_lengths)} AA "
            f"but features have {mat.shape[0]} rows"
        )

        # Build mask and extract kept rows
        keep_mask = build_keep_mask(protein_lengths, is_fwd)
        n_excluded = (~keep_mask).sum()
        print(f"    {mat.shape[0]} rows, excluding {n_excluded} edge AAs", flush=True)

        mat_kept = mat[keep_mask]

        # Per-feature max of kept rows
        # .max(axis=0) may return sparse or dense depending on scipy version
        shard_max_raw = mat_kept.max(axis=0)
        if sparse.issparse(shard_max_raw):
            shard_max = np.asarray(shard_max_raw.todense()).ravel()
        else:
            shard_max = np.asarray(shard_max_raw).ravel()
        global_max = np.maximum(global_max, shard_max)

    if global_max is None:
        print("ERROR: No shards found!", flush=True)
        return

    n_zero = (global_max == 0).sum()
    print(f"\nGlobal max stats: min={global_max.min():.4f}, "
          f"median={np.median(global_max):.4f}, max={global_max.max():.4f}, "
          f"zero_features={n_zero}", flush=True)

    # Avoid division by zero for dead features
    global_max[global_max == 0] = 1.0

    # --- Pass 2: normalize and overwrite ---
    print("\n=== Pass 2: Normalizing shards ===", flush=True)
    inv_max = (1.0 / global_max).astype(np.float32)

    for shard in range(n_shards):
        feat_path = features_dir / f"shard_{shard}_features.npz"
        if not feat_path.exists():
            continue

        print(f"  Shard {shard}: normalizing...", flush=True)
        mat = sparse.load_npz(feat_path).tocsr()

        # Efficient in-place normalization: divide each element by its column's max
        # For CSR, mat.indices gives the column index of each stored element
        mat.data = mat.data * inv_max[mat.indices]

        # Verify
        actual_max = mat.max()
        print(f"    After normalization: max={actual_max:.6f}, nnz={mat.nnz}", flush=True)

        sparse.save_npz(feat_path, mat)
        print(f"    Saved {feat_path}", flush=True)

    print("\n=== Normalization complete! ===", flush=True)


if __name__ == "__main__":
    main()
