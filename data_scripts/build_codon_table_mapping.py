#!/usr/bin/env python3
"""Parse SwissProt .dat file to assign NCBI codon tables based on taxonomy + organelle.

Writes a TSV mapping: uniprot_ac -> transl_table

Rules:
  - Bacteria/Archaea (nuclear): table 11
  - Eukaryota (nuclear): table 1
  - Viruses: table 1 (host-dependent, but standard is safest default)
  - Mitochondrion + Vertebrata: table 2
  - Mitochondrion + Saccharomyces/Kluyveromyces: table 3 (yeast mito)
  - Mitochondrion + Schizosaccharomyces: table 4
  - Mitochondrion + Arthropoda/Nematoda/Mollusca: table 5 (invertebrate mito)
  - Mitochondrion + Echinodermata/Platyhelminthes: table 9
  - Mitochondrion + Ascidia: table 13
  - Mitochondrion + other Eukaryota: table 4 (protozoan mito, most common fallback)
  - Plastid/Chloroplast: table 11 (bacterial-like)

Usage:
    python build_codon_table_mapping.py
"""

import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "uniprot"
DAT_FILE = DATA_DIR / "uniprot_sprot.dat"
OUTPUT_TSV = DATA_DIR / "uniprot_codon_tables.tsv"


def assign_codon_table(taxonomy_tokens, organelle):
    """Assign NCBI codon table based on taxonomy and organelle."""
    tax_set = set(taxonomy_tokens)

    # Plastid/chloroplast → table 11 (bacterial-like translation)
    if organelle and ("Plastid" in organelle or "Chloroplast" in organelle or
                      "Cyanelle" in organelle or "chromatophore" in organelle.lower()):
        return 11

    # Mitochondrial
    if organelle and "Mitochondrion" in organelle:
        if "Vertebrata" in tax_set or "Craniata" in tax_set:
            return 2
        if "Saccharomyces" in tax_set or "Kluyveromyces" in tax_set:
            return 3
        if "Schizosaccharomyces" in tax_set:
            return 4
        if tax_set & {"Arthropoda", "Nematoda", "Mollusca", "Annelida", "Insecta"}:
            return 5
        if tax_set & {"Echinodermata", "Platyhelminthes"}:
            return 9
        if "Ascidia" in tax_set or "Ascidiacea" in tax_set:
            return 13
        # Other eukaryotic mitochondria (protozoa, plants, fungi)
        return 4

    # Nuclear / non-organellar
    if "Bacteria" in tax_set or "Archaea" in tax_set:
        return 11
    if "Eukaryota" in tax_set:
        return 1
    if "Viruses" in tax_set:
        return 1

    # Fallback
    return 1


def main():
    print(f"Parsing {DAT_FILE}...", flush=True)

    results = []
    current_ac = None
    oc_tokens = []
    organelle = None
    got_embl = False  # only output entries that have EMBL cross-refs

    with open(DAT_FILE) as f:
        for line in f:
            if line.startswith("AC   ") and current_ac is None:
                current_ac = line[5:].strip().rstrip(";").split(";")[0].strip()
            elif line.startswith("OC   "):
                # OC   Eukaryota; Metazoa; Chordata; Craniata; Vertebrata; ...
                tokens = [t.strip().rstrip(".") for t in line[5:].split(";")]
                oc_tokens.extend(t for t in tokens if t)
            elif line.startswith("OG   "):
                organelle = line[5:].strip().rstrip(".")
            elif line.startswith("DR   EMBL;"):
                got_embl = True
            elif line.startswith("//"):
                if current_ac and got_embl:
                    table = assign_codon_table(oc_tokens, organelle)
                    results.append((current_ac, table, organelle or ""))
                current_ac = None
                oc_tokens = []
                organelle = None
                got_embl = False

    print(f"Parsed {len(results):,} entries with EMBL cross-refs", flush=True)

    # Write output
    with open(OUTPUT_TSV, "w") as f:
        f.write("uniprot_ac\ttransl_table\torganelle\n")
        for ac, table, org in results:
            f.write(f"{ac}\t{table}\t{org}\n")

    print(f"Written to {OUTPUT_TSV}", flush=True)

    # Summary
    from collections import Counter
    table_counts = Counter(t for _, t, _ in results)
    print(f"\nCodon table distribution:")
    for table, count in table_counts.most_common():
        print(f"  table {table:>2d}: {count:>8,}")


if __name__ == "__main__":
    main()
