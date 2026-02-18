#!/bin/bash
# Download all Swiss-Prot annotations TSV from UniProt REST API
# Fields match what InterPLM's extract_annotations.py expects
# Based on README Option B, with query filters removed to get all of Swiss-Prot

DATA_DIR="$(cd "$(dirname "$0")/../data/uniprotkb" && pwd)"
mkdir -p "$DATA_DIR"

wget --no-check-certificate -O "${DATA_DIR}/swissprot_annotations.tsv.gz" \
  "https://rest.uniprot.org/uniprotkb/stream?compressed=true&fields=accession%2Creviewed%2Cprotein_name%2Clength%2Csequence%2Cec%2Cft_act_site%2Cft_binding%2Ccc_cofactor%2Cft_disulfid%2Cft_carbohyd%2Cft_lipid%2Cft_mod_res%2Cft_signal%2Cft_transit%2Cft_helix%2Cft_turn%2Cft_strand%2Cft_coiled%2Ccc_domain%2Cft_compbias%2Cft_domain%2Cft_motif%2Cft_region%2Cft_zn_fing%2Cxref_alphafolddb&format=tsv&query=%28reviewed%3Atrue%29"

echo "Downloaded to ${DATA_DIR}/swissprot_annotations.tsv.gz"
