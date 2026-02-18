#!/usr/bin/env python3
"""Download UniProtKB annotations TSV for the 50K sampled proteins.

Uses the UniProt REST API ID mapping endpoint to fetch structured
annotations in the format expected by InterPLM's extract_annotations.py.

Usage:
    python download_swissprot_annotations.py
"""

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "uniprot"
MAPPING_TSV = DATA_DIR / "swissprot_cds_sample_50k_mapping.tsv"
OUTPUT_TSV = DATA_DIR / "swissprot_sample_50k_annotations.tsv"

# UniProt REST API endpoints
IDMAPPING_RUN = "https://rest.uniprot.org/idmapping/run"
IDMAPPING_STATUS = "https://rest.uniprot.org/idmapping/status/{}"
IDMAPPING_RESULTS = "https://rest.uniprot.org/idmapping/uniprotkb/results/stream/{}"

# Fields to download (must match what extract_annotations.py expects)
FIELDS = ",".join([
    "accession",
    "sequence",
    "length",
    "xref_alphafolddb",
    # Categorical features
    "ft_domain",
    "ft_act_site",
    "ft_binding",
    "cc_cofactor",
    "ft_carbohyd",
    "ft_mod_res",
    "ft_region",
    "ft_motif",
    "ft_zn_fing",
    "ft_signal",
    "ft_transit",
    "ft_compbias",
    # Binary features
    "ft_helix",
    "ft_strand",
    "ft_turn",
    "ft_coiled",
    "ft_lipid",
    # Interaction features
    "ft_disulfid",
])


def submit_id_mapping(accessions):
    """Submit ID mapping job to UniProt."""
    data = urllib.parse.urlencode({
        "from": "UniProtKB_AC-ID",
        "to": "UniProtKB",
        "ids": ",".join(accessions),
    }).encode()
    req = urllib.request.Request(IDMAPPING_RUN, data=data)
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read().decode())
    return result["jobId"]


def poll_status(job_id, interval=5, max_wait=600):
    """Poll until the ID mapping job completes."""
    url = IDMAPPING_STATUS.format(job_id)
    start = time.time()
    while time.time() - start < max_wait:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as resp:
            # Follow redirects — if we get results URL, job is done
            if resp.url != url:
                return True
            result = json.loads(resp.read().decode())
            if "jobStatus" in result:
                status = result["jobStatus"]
                print(f"  Job status: {status}", flush=True)
                if status == "FINISHED":
                    return True
                if status in ("FAILED", "ERROR"):
                    raise RuntimeError(f"ID mapping job failed: {result}")
            else:
                # No jobStatus field means results are ready
                return True
        time.sleep(interval)
    raise TimeoutError(f"Job {job_id} did not complete within {max_wait}s")


def download_results(job_id):
    """Download TSV results from completed ID mapping job."""
    url = IDMAPPING_RESULTS.format(job_id)
    params = urllib.parse.urlencode({
        "format": "tsv",
        "fields": FIELDS,
    })
    full_url = f"{url}?{params}"
    req = urllib.request.Request(full_url)
    with urllib.request.urlopen(req) as resp:
        return resp.read().decode()


def main():
    # Load sample accessions
    print("Loading sample accessions...", flush=True)
    accessions = []
    with open(MAPPING_TSV) as f:
        header = f.readline().strip().split("\t")
        ac_col = header.index("uniprot_ac")
        for line in f:
            accessions.append(line.strip().split("\t")[ac_col])
    # Deduplicate while preserving order
    seen = set()
    unique_acs = []
    for ac in accessions:
        if ac not in seen:
            seen.add(ac)
            unique_acs.append(ac)
    print(f"  {len(unique_acs):,} unique accessions", flush=True)

    # Submit ID mapping (50K fits in one job, limit is 100K)
    print("Submitting ID mapping job...", flush=True)
    job_id = submit_id_mapping(unique_acs)
    print(f"  Job ID: {job_id}", flush=True)

    # Poll for completion
    print("Waiting for job to complete...", flush=True)
    poll_status(job_id)
    print("  Job completed!", flush=True)

    # Download results
    print("Downloading TSV results...", flush=True)
    tsv_data = download_results(job_id)
    lines = tsv_data.strip().split("\n")
    print(f"  Got {len(lines) - 1:,} rows (+ header)", flush=True)

    # Write output
    print(f"Writing {OUTPUT_TSV}...", flush=True)
    with open(OUTPUT_TSV, "w") as f:
        f.write(tsv_data)

    # Summary
    print(f"\nDone! Output: {OUTPUT_TSV}")
    print(f"  Rows: {len(lines) - 1:,}")
    print(f"  Columns: {lines[0].count(chr(9)) + 1}")
    print(f"  Header: {lines[0][:200]}...")


if __name__ == "__main__":
    main()
