## Evo2 SAE Concept Evaluation

### Overview

We evaluated whether sparse autoencoder (SAE) features learned from Evo2 DNA language model embeddings can detect protein-level functional annotations from UniProt/SwissProt. This extends the InterPLM concept evaluation framework — originally designed for protein language models (ESM2) — to a genomic foundation model operating at nucleotide resolution.

### Models Evaluated

- **Evo2-7B**: SAE with 65,536 features trained on layer 26 embeddings (TopK SAE, k=64)
- **Evo2-40B**: SAE with 131,072 features trained on layer 18 embeddings (TopK SAE, k=64)
- Both evaluated on forward and reverse complement strands independently
- Baselines: ESM2-8M (10,240 features, layer 4) and ESM2-650M (10,240 features, layer 24)

### Dataset

Pre-computed Evo2 SAE feature activations for ~50,000 SwissProt CDS sequences (a superset of the 46,165 proteins with SwissProt annotations used in the evaluation). Annotations span 584 non-amino-acid concepts (protein domains, motifs, binding sites, etc.) across 8 shards, split into validation (shards 0–3) and test (shards 4–7) sets.

### Resolution Alignment: Codon Mean-Pooling

Evo2 SAE features are at nucleotide resolution while SwissProt annotations are at amino acid resolution. We aligned these by mean-pooling each feature across 3-nucleotide codons:

1. For each CDS, loaded the sparse nucleotide-level feature matrix (CSR, float16 → float32)
2. Trimmed trailing nucleotides not forming complete codons (230/50k sequences affected)
3. Detected and discarded stop codons when present (99.2% of sequences include one; for forward strand, last 3 nt are discarded; for reverse complement, first 3 nt)
4. Reshaped to (n_aa, 3, n_features) and averaged across the codon axis
5. For reverse complement: reversed row order after pooling to align with forward-strand amino acid ordering

Protein-to-CDS mapping used the CDS protein accession as the sole key (resolving 301/50k cases where EMBL nucleotide accessions differed between the UniProt mapping and actual NCBI records). Translation correctness was verified for all 50,000 sequences using BioPython with organism-specific codon tables.

### Feature Normalization

Evo2 SAE activations have substantially larger magnitudes than ESM2 (Evo2-7B median: 0.32, 99th percentile: 3.24; Evo2-40B median: 1.06, 99th percentile: 10.21; vs. ESM2-8M median: 0.06, 99th percentile: 0.58). Since the comparison pipeline binarizes features at absolute thresholds [0, 0.15, 0.5, 0.6, 0.8], we normalized Evo2 features by dividing each feature by its global maximum across all 8 shards. To avoid bias from anomalously large activations at sequence boundaries (a known property of Evo2 embeddings), we excluded the first 10 amino acids per protein (forward strand) or last 10 amino acids (reverse complement) when computing the per-feature maximum.

### Evaluation Pipeline

After caching normalized features, we ran the standard InterPLM concept evaluation:
1. **Compare**: For each (concept, feature, threshold) triple, computed true positives, false positives, and true positives per domain via sparse matrix multiplication
2. **F1 scoring**: Aggregated counts across shards and computed per-concept F1 scores (both per-token and per-domain)
3. **Heldout evaluation**: Selected best feature–concept pairings on the validation set, reported F1 on the held-out test set

### Results

| Model | Concepts Evaluated | Identified (F1 ≥ 0.5) | Mean F1 | Median F1 |
|---|---|---|---|---|
| ESM2-8M | 531 | 143 | 0.288 | 0.119 |
| ESM2-650M | 569 | 238 | 0.437 | 0.383 |
| Evo2-7B (fwd) | 424 | 75 | 0.243 | 0.123 |
| Evo2-7B (revcomp) | 423 | 90 | 0.259 | 0.138 |
| Evo2-40B (fwd) | 439 | 164 | 0.381 | 0.294 |
| Evo2-40B (revcomp) | 452 | 175 | 0.390 | 0.303 |

Evo2-40B SAE features detect protein-level concepts at a level approaching ESM2-650M, despite operating on raw DNA sequence rather than protein sequence. The reverse complement strand performs slightly better than forward for both Evo2 models. Evo2-7B performs comparably to ESM2-8M. Scaling from 7B to 40B roughly doubles the number of identified concepts, mirroring the ESM2-8M → ESM2-650M scaling trend.

### Implementation

All scripts are in `autointerp/InterPLM/data_scripts/`:
- `cache_evo2_shard_features.py` — Codon mean-pooling and shard assembly
- `normalize_evo2_shard_features.py` — Per-feature global max normalization
- `eval_evo2_cache.sh` — SLURM array job for caching (8 shards × 4 configs)
- `eval_evo2_normalize.sh` — SLURM job for normalization
- `eval_evo2_compare.sh` — SLURM array job for TP/FP comparison (16 tasks × 4 configs)
- `eval_evo2_f1.sh` — SLURM array job for F1 aggregation
- `eval_evo2_report.sh` — Final metrics reporting
