#!/usr/bin/env python3
"""Download CDS nucleotide sequences for SwissProt entries from NCBI Entrez.

Parses UniProt .dat file to extract EMBL CDS protein accessions,
then batch-fetches CDS nucleotide sequences via NCBI efetch.

Designed to be run as a SLURM array job: each task processes a shard
of accessions. Preemption-safe via .incomplete suffix pattern.

Usage:
    python download_swissprot_cds.py --shard-id 0 --num-shards 50
    python download_swissprot_cds.py --shard-id 0 --num-shards 50 --dry-run
"""

import argparse
import os
import re
import signal
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "")
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
BATCH_SIZE = 200
# With API key: 10 requests/sec; stay conservative
REQUEST_DELAY = 0.15
MAX_RETRIES = 5

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DAT_FILE = DATA_DIR / "uniprot" / "uniprot_sprot.dat"
OUTPUT_DIR = DATA_DIR / "uniprot" / "cds_shards"

# Graceful shutdown
shutdown_requested = False


def handle_signal(signum, frame):
    global shutdown_requested
    print(f"Received signal {signum} at {time.strftime('%Y-%m-%d %H:%M:%S')}, finishing current batch...", flush=True)
    shutdown_requested = True


signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)


def parse_dat_file(dat_path):
    """Parse UniProt .dat file, return list of (uniprot_ac, embl_nucl_acc, cds_protein_acc, mol_type).

    Picks one CDS accession per UniProt entry (first non-suppressed EMBL cross-ref).
    """
    entries = []
    current_ac = None

    with open(dat_path) as f:
        for line in f:
            if line.startswith("AC   "):
                # First AC line for entry; take first accession
                current_ac = line[5:].strip().rstrip(";").split(";")[0].strip()
            elif line.startswith("DR   EMBL;") and current_ac is not None:
                # DR   EMBL; AY548484; AAT09660.1; -; Genomic_DNA.
                parts = [p.strip().rstrip(".") for p in line[5:].split(";")]
                if len(parts) >= 4:
                    nucl_acc = parts[1].strip()
                    protein_acc = parts[2].strip()
                    mol_type = parts[4].strip() if len(parts) >= 5 else ""
                    # Skip suppressed/missing accessions
                    if protein_acc and protein_acc != "-":
                        entries.append((current_ac, nucl_acc, protein_acc, mol_type))
                        current_ac = None  # Only take first per entry
            elif line.startswith("//"):
                current_ac = None  # Reset for next entry

    return entries


def parse_fasta_headers(fasta_text):
    """Extract metadata from NCBI fasta_cds_na headers.

    Returns dict mapping protein_id -> {protein_id, transl_table, location, gene, product}.
    Header format: >lcl|... [protein_id=X] [transl_table=11] [location=...] [gene=...] ...
    """
    metadata = {}
    for line in fasta_text.splitlines():
        if not line.startswith(">"):
            continue
        fields = {}
        for m in re.finditer(r'\[(\w+)=([^\]]*)\]', line):
            fields[m.group(1)] = m.group(2)
        pid = fields.get("protein_id", "")
        if pid:
            metadata[pid] = {
                "protein_id": pid,
                "transl_table": fields.get("transl_table", "1"),  # default = standard
                "location": fields.get("location", ""),
                "gene": fields.get("gene", ""),
                "protein": fields.get("protein", ""),
            }
    return metadata


def fetch_cds_batch(protein_accessions):
    """Fetch CDS nucleotide sequences from NCBI for a batch of protein accessions.

    Returns FASTA text or raises on failure.
    """
    params = urllib.parse.urlencode({
        "db": "protein",
        "id": ",".join(protein_accessions),
        "rettype": "fasta_cds_na",
        "retmode": "text",
        "api_key": NCBI_API_KEY,
    })
    url = f"{EFETCH_URL}?{params}"

    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            wait = 2 ** attempt + 1
            print(f"  Retry {attempt+1}/{MAX_RETRIES} after error: {e} (waiting {wait}s)", flush=True)
            time.sleep(wait)

    raise RuntimeError(f"Failed after {MAX_RETRIES} retries for {len(protein_accessions)} accessions")


def main():
    parser = argparse.ArgumentParser(description="Download SwissProt CDS nucleotide sequences")
    parser.add_argument("--shard-id", type=int, required=True, help="Shard index (0-based)")
    parser.add_argument("--num-shards", type=int, required=True, help="Total number of shards")
    parser.add_argument("--dat-file", type=str, default=str(DAT_FILE))
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--dry-run", action="store_true", help="Parse and shard only, don't fetch")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"swissprot_cds_{args.shard_id:04d}.fasta"
    mapping_file = output_dir / f"swissprot_cds_{args.shard_id:04d}.tsv"
    progress_file = output_dir / f"swissprot_cds_{args.shard_id:04d}.progress"

    # Check if already completed
    if output_file.exists() and mapping_file.exists() and not Path(str(output_file) + ".incomplete").exists():
        print(f"Shard {args.shard_id} already completed: {output_file}", flush=True)
        return

    # Parse .dat file
    print(f"Parsing {args.dat_file}...", flush=True)
    entries = parse_dat_file(args.dat_file)
    print(f"Parsed {len(entries):,} UniProt entries with CDS accessions", flush=True)

    # Shard
    shard_entries = [e for i, e in enumerate(entries) if i % args.num_shards == args.shard_id]
    print(f"Shard {args.shard_id}/{args.num_shards}: {len(shard_entries):,} entries", flush=True)

    if args.dry_run:
        print("Dry run — exiting.", flush=True)
        return

    # Check progress from previous run (preemption recovery)
    completed_batches = 0
    if progress_file.exists():
        completed_batches = int(progress_file.read_text().strip())
        print(f"Resuming from batch {completed_batches}", flush=True)

    # Build lookup: protein_acc -> (uniprot_ac, embl_nucl_acc, mol_type)
    acc_to_uniprot = {
        protein_acc: (uniprot_ac, nucl_acc, mol_type)
        for uniprot_ac, nucl_acc, protein_acc, mol_type in shard_entries
    }

    # Batch fetch
    total_batches = (len(shard_entries) + BATCH_SIZE - 1) // BATCH_SIZE
    incomplete_path = str(output_file) + ".incomplete"
    mapping_incomplete = str(mapping_file) + ".incomplete"
    fasta_mode = "a" if completed_batches > 0 else "w"
    tsv_mode = "a" if completed_batches > 0 else "w"
    sequences_written = 0

    with open(incomplete_path, fasta_mode) as fout, open(mapping_incomplete, tsv_mode) as tsvout:
        if completed_batches == 0:
            tsvout.write("uniprot_ac\tembl_nucl_acc\tcds_protein_acc\tmol_type\ttransl_table\tgene\tproduct\n")

        for batch_idx in range(completed_batches, total_batches):
            if shutdown_requested:
                print(f"Shutdown requested, stopping after batch {batch_idx}", flush=True)
                progress_file.write_text(str(batch_idx))
                print(f"Progress saved: {batch_idx}/{total_batches} batches", flush=True)
                sys.exit(143)

            start = batch_idx * BATCH_SIZE
            end = min(start + BATCH_SIZE, len(shard_entries))
            batch = shard_entries[start:end]
            protein_accs = [acc for _, _, acc, _ in batch]

            try:
                fasta_text = fetch_cds_batch(protein_accs)
                if fasta_text.strip():
                    fout.write(fasta_text)
                    if not fasta_text.endswith("\n"):
                        fout.write("\n")
                    n_seqs = fasta_text.count(">")
                    sequences_written += n_seqs

                    # Parse headers and write mapping rows
                    header_meta = parse_fasta_headers(fasta_text)
                    for pid, meta in header_meta.items():
                        if pid in acc_to_uniprot:
                            uniprot_ac, nucl_acc, mol_type = acc_to_uniprot[pid]
                        else:
                            uniprot_ac, nucl_acc, mol_type = "", "", ""
                        tsvout.write(f"{uniprot_ac}\t{nucl_acc}\t{pid}\t{mol_type}\t"
                                     f"{meta['transl_table']}\t{meta['gene']}\t{meta['protein']}\n")
                else:
                    print(f"  WARNING: Empty response for batch {batch_idx} ({len(protein_accs)} accessions)", flush=True)
            except RuntimeError as e:
                print(f"  ERROR: {e} — skipping batch {batch_idx}", flush=True)

            # Flush and save progress periodically
            if (batch_idx + 1) % 10 == 0:
                fout.flush()
                tsvout.flush()
                progress_file.write_text(str(batch_idx + 1))

            if (batch_idx + 1) % 50 == 0 or batch_idx == total_batches - 1:
                print(f"  Batch {batch_idx+1}/{total_batches} "
                      f"({100*(batch_idx+1)/total_batches:.1f}%), "
                      f"{sequences_written:,} sequences written", flush=True)

            time.sleep(REQUEST_DELAY)

    # Completed successfully — rename
    os.rename(incomplete_path, str(output_file))
    os.rename(mapping_incomplete, str(mapping_file))
    if progress_file.exists():
        progress_file.unlink()
    print(f"Shard {args.shard_id} complete: {output_file} ({sequences_written:,} sequences)", flush=True)


if __name__ == "__main__":
    main()
