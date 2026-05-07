# The GENETIC_CODE, BASES, and COMP constants below are hardcoded lookup tables
# used to translate codons to amino acids and complement nucleotides. These are
# not imported from a library, they are defined explicitly here so the script
# has no extra external dependencies.
# Allows for consequence classification (synonymous, nonsynonymous,
# nonsense) for every possible single-base substitution at every CDS position.

import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import numpy as np

# Settings
VCF_DIR = Path("/shared/team/2025-masters-project/people/justin/all_genomes_singletonsandwindowmasked")
ANNOT_DIR = Path("/shared/team/2025-masters-project/people/justin/all_genomes_referenceandFFA_files")

VCF_SUFFIX = "_biallelic_k1_singletons_windowmasked.vcf"
FASTA_SUFFIX = ".fna"
GFF_SUFFIX = ".gff"

ALLOW_FILTER = {"PASS", "."}
SNP_ONLY_BIALLELIC = True
ENFORCE_K1_AFTER_MASKING = True
SKIP_ALL_REF_ROWS = False

FEATURE_TYPES = {"CDS"}
COLORMAP_NAME = "plasma"

BASES = {"A", "C", "G", "T"}
COMP = {"A": "T", "T": "A", "C": "G", "G": "C"}

GENETIC_CODE = {
    "TTT":"F","TTC":"F","TTA":"L","TTG":"L",
    "TCT":"S","TCC":"S","TCA":"S","TCG":"S",
    "TAT":"Y","TAC":"Y","TAA":"*","TAG":"*",
    "TGT":"C","TGC":"C","TGA":"*","TGG":"W",
    "CTT":"L","CTC":"L","CTA":"L","CTG":"L",
    "CCT":"P","CCC":"P","CCA":"P","CCG":"P",
    "CAT":"H","CAC":"H","CAA":"Q","CAG":"Q",
    "CGT":"R","CGC":"R","CGA":"R","CGG":"R",
    "ATT":"I","ATC":"I","ATA":"I","ATG":"M",
    "ACT":"T","ACC":"T","ACA":"T","ACG":"T",
    "AAT":"N","AAC":"N","AAA":"K","AAG":"K",
    "AGT":"S","AGC":"S","AGA":"R","AGG":"R",
    "GTT":"V","GTC":"V","GTA":"V","GTG":"V",
    "GCT":"A","GCC":"A","GCA":"A","GCG":"A",
    "GAT":"D","GAC":"D","GAA":"E","GAG":"E",
    "GGT":"G","GGC":"G","GGA":"G","GGG":"G"
}

# Species label formatting
def format_species_label(label):
    label = str(label).strip().replace("_", " ")
    parts = label.split()
    if not parts:
        return ""
    return " ".join([parts[0].capitalize()] + [p.lower() for p in parts[1:]])

def apply_species_xticklabels(ax, labels, rotation=60, fontsize=9):
    formatted = [format_species_label(lbl) for lbl in labels]
    ax.set_xticklabels(
        formatted,
        rotation=rotation,
        ha="right",
        rotation_mode="anchor",
        fontsize=fontsize
    )
    for tick in ax.get_xticklabels():
        tick.set_fontstyle("italic")

def ansi_italic(text):
    return f"\x1b[3m{text}\x1b[0m"

# FASTA
def read_fasta_dict(fasta_path: str) -> dict:
    seqs = {}
    name = None
    chunks = []
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    seqs[name] = "".join(chunks).upper()
                name = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
        if name is not None:
            seqs[name] = "".join(chunks).upper()
    return seqs

# GFF / CDS
def parse_gff_cds_features(gff_path: str, feature_types={"CDS"}):
    cds_features = []
    with open(gff_path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            seqid = parts[0]
            feature_type = parts[2]
            start = parts[3]
            end = parts[4]
            strand = parts[6]
            phase = parts[7]
            if feature_type not in feature_types:
                continue
            if strand not in {"+", "-"}:
                continue
            try:
                start = int(start)
                end = int(end)
            except ValueError:
                continue
            if start > end:
                start, end = end, start
            try:
                phase = int(phase)
            except ValueError:
                phase = 0
            if phase not in {0, 1, 2}:
                phase = 0
            cds_features.append((seqid, start, end, strand, phase))
    return cds_features

def revcomp(seq: str) -> str:
    return "".join(COMP.get(b, "N") for b in reversed(seq.upper()))

# Build per-base CDS consequence map
def build_effect_maps(ref_seqs, cds_features):
    assignments = defaultdict(lambda: defaultdict(list))
    opportunity_counts = {"synonymous": 0, "nonsynonymous": 0, "nonsense": 0}

    for seqid, start, end, strand, phase in cds_features:
        if seqid not in ref_seqs:
            continue
        seq = ref_seqs[seqid]
        n = len(seq)
        start = max(1, start)
        end = min(n, end)
        if start > end:
            continue
        if strand == "+":
            genomic_positions = list(range(start, end + 1))
        else:
            genomic_positions = list(range(end, start - 1, -1))
        genomic_positions = genomic_positions[phase:]
        usable_len = (len(genomic_positions) // 3) * 3
        genomic_positions = genomic_positions[:usable_len]

        for i in range(0, len(genomic_positions), 3):
            codon_positions = genomic_positions[i:i+3]
            if len(codon_positions) != 3:
                continue
            genomic_triplet = "".join(seq[pos - 1].upper() for pos in codon_positions)
            if any(b not in BASES for b in genomic_triplet):
                continue
            if strand == "+":
                codon = genomic_triplet
            else:
                codon = revcomp(genomic_triplet)
            if codon not in GENETIC_CODE:
                continue
            aa_ref = GENETIC_CODE[codon]

            for codon_idx, genome_pos in enumerate(codon_positions):
                ref_base_coding = codon[codon_idx]
                for alt_base_coding in BASES:
                    if alt_base_coding == ref_base_coding:
                        continue
                    mutant = list(codon)
                    mutant[codon_idx] = alt_base_coding
                    mutant = "".join(mutant)
                    if mutant not in GENETIC_CODE:
                        continue
                    aa_alt = GENETIC_CODE[mutant]
                    if aa_alt == aa_ref:
                        effect = "synonymous"
                    elif aa_alt == "*":
                        effect = "nonsense"
                    else:
                        effect = "nonsynonymous"
                    if strand == "+":
                        genomic_alt = alt_base_coding
                    else:
                        genomic_alt = COMP[alt_base_coding]
                    assignments[seqid][genome_pos].append((genomic_alt, effect))
                    opportunity_counts[effect] += 1

    effect_map = {}
    ambiguous_positions = 0

    for seqid in ref_seqs:
        effect_map[seqid] = {}
        pos_dict = assignments.get(seqid, {})
        for pos, entries in pos_dict.items():
            alt_to_effects = defaultdict(set)
            for alt, effect in entries:
                alt_to_effects[alt].add(effect)
            if any(len(effects) > 1 for effects in alt_to_effects.values()):
                ambiguous_positions += 1
                continue
            effect_map[seqid][pos] = {
                alt: list(effects)[0] for alt, effects in alt_to_effects.items()
            }

    return effect_map, opportunity_counts, ambiguous_positions

# VCF helpers
def gt_index_from_format(fmt: str):
    fields = fmt.split(":")
    return fields.index("GT") if "GT" in fields else None

def is_alt_carrier(sample_field: str, gt_index: int) -> bool:
    parts = sample_field.split(":")
    if gt_index is None or gt_index >= len(parts):
        return False
    gt = parts[gt_index]
    if gt in {".", "./.", ".|."}:
        return False
    alleles = gt.replace("|", "/").split("/")
    return "1" in alleles

def is_pass_snp_row(col) -> bool:
    if len(col) > 6 and col[6] not in ALLOW_FILTER:
        return False
    ref = col[3].upper()
    alt = col[4].upper()
    if SNP_ONLY_BIALLELIC:
        if "," in alt:
            return False
        if ref not in BASES or alt not in BASES:
            return False
        if ref == alt:
            return False
    return True

def count_alt_carriers(sample_fields, gt_index: int) -> int:
    c = 0
    for sf in sample_fields:
        if is_alt_carrier(sf, gt_index):
            c += 1
            if c > 1 and ENFORCE_K1_AFTER_MASKING:
                break
    return c

def species_name_from_vcf(vcf_path: Path) -> str:
    if vcf_path.name.endswith(VCF_SUFFIX):
        return vcf_path.name[:-len(VCF_SUFFIX)]
    return vcf_path.stem

# Analyse one species
def analyse_species_syn_non_nonsense(vcf_path: Path, fasta_path: Path, gff_path: Path):
    ref_seqs = read_fasta_dict(fasta_path)
    cds_features = parse_gff_cds_features(gff_path, feature_types=FEATURE_TYPES)
    effect_map, opportunity_counts, ambiguous_positions = build_effect_maps(ref_seqs, cds_features)

    obs = {"synonymous": 0, "nonsynonymous": 0, "nonsense": 0}

    total_rows = 0
    used_rows = 0
    skipped_all_ref = 0
    skipped_not_k1 = 0
    skipped_missing_seq = 0
    skipped_bad_pos = 0
    skipped_noncoding = 0
    skipped_unmapped_alt = 0

    with open(vcf_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            total_rows += 1
            col = line.rstrip("\n").split("\t")
            if not is_pass_snp_row(col):
                continue
            gt_idx = gt_index_from_format(col[8])
            if gt_idx is None:
                continue
            samples = col[9:]
            carriers = count_alt_carriers(samples, gt_idx)
            if carriers == 0 and SKIP_ALL_REF_ROWS:
                skipped_all_ref += 1
                continue
            if ENFORCE_K1_AFTER_MASKING and carriers != 1:
                skipped_not_k1 += 1
                continue
            chrom = col[0]
            if chrom not in ref_seqs or chrom not in effect_map:
                skipped_missing_seq += 1
                continue
            try:
                pos = int(col[1])
            except ValueError:
                skipped_bad_pos += 1
                continue
            if pos < 1 or pos > len(ref_seqs[chrom]):
                skipped_bad_pos += 1
                continue
            alt = col[4].upper()
            if alt not in BASES:
                continue
            if pos not in effect_map[chrom]:
                skipped_noncoding += 1
                continue
            alt_map = effect_map[chrom][pos]
            if alt not in alt_map:
                skipped_unmapped_alt += 1
                continue
            effect = alt_map[alt]
            obs[effect] += 1
            used_rows += 1

    raw_total = obs["synonymous"] + obs["nonsynonymous"] + obs["nonsense"]
    raw_syn_prop = obs["synonymous"] / raw_total if raw_total > 0 else 0.0
    raw_non_prop = obs["nonsynonymous"] / raw_total if raw_total > 0 else 0.0
    raw_stop_prop = obs["nonsense"] / raw_total if raw_total > 0 else 0.0

    norm_syn = obs["synonymous"] / opportunity_counts["synonymous"] if opportunity_counts["synonymous"] > 0 else 0.0
    norm_non = obs["nonsynonymous"] / opportunity_counts["nonsynonymous"] if opportunity_counts["nonsynonymous"] > 0 else 0.0
    norm_stop = obs["nonsense"] / opportunity_counts["nonsense"] if opportunity_counts["nonsense"] > 0 else 0.0

    norm_total = norm_syn + norm_non + norm_stop
    norm_syn_prop = norm_syn / norm_total if norm_total > 0 else 0.0
    norm_non_prop = norm_non / norm_total if norm_total > 0 else 0.0
    norm_stop_prop = norm_stop / norm_total if norm_total > 0 else 0.0

    ratio_syn_non = norm_syn / norm_non if norm_non > 0 else float("inf")
    ratio_syn_stop = norm_syn / norm_stop if norm_stop > 0 else float("inf")
    ratio_non_stop = norm_non / norm_stop if norm_stop > 0 else float("inf")

    dn_ds_ratio = norm_non / norm_syn if norm_syn > 0 else float("inf")

    return {
        "total_rows": total_rows,
        "used_rows": used_rows,
        "skipped_all_ref": skipped_all_ref,
        "skipped_not_k1": skipped_not_k1,
        "skipped_missing_seq": skipped_missing_seq,
        "skipped_bad_pos": skipped_bad_pos,
        "skipped_noncoding": skipped_noncoding,
        "skipped_unmapped_alt": skipped_unmapped_alt,
        "ambiguous_positions": ambiguous_positions,
        "opp_synonymous": opportunity_counts["synonymous"],
        "opp_nonsynonymous": opportunity_counts["nonsynonymous"],
        "opp_nonsense": opportunity_counts["nonsense"],
        "raw_synonymous": obs["synonymous"],
        "raw_nonsynonymous": obs["nonsynonymous"],
        "raw_nonsense": obs["nonsense"],
        "raw_syn_prop": raw_syn_prop,
        "raw_non_prop": raw_non_prop,
        "raw_stop_prop": raw_stop_prop,
        "norm_synonymous": norm_syn,
        "norm_nonsynonymous": norm_non,
        "norm_nonsense": norm_stop,
        "norm_syn_prop": norm_syn_prop,
        "norm_non_prop": norm_non_prop,
        "norm_stop_prop": norm_stop_prop,
        "ratio_syn_non": ratio_syn_non,
        "ratio_syn_stop": ratio_syn_stop,
        "ratio_non_stop": ratio_non_stop,
        "dn_ds_ratio": dn_ds_ratio,
    }

# Run all species
vcf_files = sorted(VCF_DIR.glob(f"*{VCF_SUFFIX}"))
results = {}

for vcf_path in vcf_files:
    species = species_name_from_vcf(vcf_path)
    fasta_path = ANNOT_DIR / f"{species}{FASTA_SUFFIX}"
    gff_path = ANNOT_DIR / f"{species}{GFF_SUFFIX}"

    if not fasta_path.exists():
        print(f"Skipping {species}: missing FASTA {fasta_path.name}")
        continue
    if not gff_path.exists():
        print(f"Skipping {species}: missing GFF {gff_path.name}")
        continue

    res = analyse_species_syn_non_nonsense(vcf_path, fasta_path, gff_path)
    results[species] = res

    print(f"Processed {species}")
    print(f"  norm synonymous rate     = {res['norm_synonymous']:.6e}")
    print(f"  norm nonsynonymous rate  = {res['norm_nonsynonymous']:.6e}")
    print(f"  norm nonsense rate       = {res['norm_nonsense']:.6e}")
    print(f"  dN/dS-like ratio         = {res['dn_ds_ratio']:.6f}")

if len(results) == 0:
    raise ValueError("No species were processed. Check paths and filename matching.")

# Species ordering
species_names = sorted(results.keys())
display_names = [format_species_label(s) for s in species_names]

species_names_d = sorted(species_names, key=lambda s: results[s]["dn_ds_ratio"], reverse=True)
display_names_d = [format_species_label(s) for s in species_names_d]

# Table
print("\nSynonymous / nonsynonymous / nonsense summary table:\n")

SPECIES_WIDTH = max(len(format_species_label(s)) for s in species_names) + 2

header = (
    f"{'Species':<{SPECIES_WIDTH}}"
    f"{'Raw_syn':>10}"
    f"{'Raw_non':>10}"
    f"{'Raw_stop':>10}"
    f"{'Opp_syn':>12}"
    f"{'Opp_non':>12}"
    f"{'Opp_stop':>12}"
    f"{'Norm_syn':>14}"
    f"{'Norm_non':>14}"
    f"{'Norm_stop':>14}"
    f"{'dN/dS':>12}"
)
print(header)
print("-" * len(header))

def fmt_ratio(x):
    return f"{x:.4f}" if np.isfinite(x) else "inf"

for s in species_names:
    r = results[s]
    species_label = format_species_label(s)
    print(
        f"{species_label:<{SPECIES_WIDTH}}"
        f"{r['raw_synonymous']:10d}"
        f"{r['raw_nonsynonymous']:10d}"
        f"{r['raw_nonsense']:10d}"
        f"{r['opp_synonymous']:12d}"
        f"{r['opp_nonsynonymous']:12d}"
        f"{r['opp_nonsense']:12d}"
        f"{r['norm_synonymous']:14.6e}"
        f"{r['norm_nonsynonymous']:14.6e}"
        f"{r['norm_nonsense']:14.6e}"
        f"{fmt_ratio(r['dn_ds_ratio']):>12}"
    )

# Colours
cmap = plt.colormaps[COLORMAP_NAME]
n_species = len(species_names)

species_colors = {
    species: cmap(0.35 + 0.5 * (i / (n_species - 1 if n_species > 1 else 1)))
    for i, species in enumerate(species_names)
}
bar_colors = [species_colors[s] for s in species_names]
bar_colors_d = [species_colors[s] for s in species_names_d]

# Values for plotting
syn_vals = [results[s]["norm_syn_prop"] for s in species_names]
non_vals = [results[s]["norm_non_prop"] for s in species_names]
stop_vals = [results[s]["norm_stop_prop"] for s in species_names]
x_abc = np.arange(len(species_names))

dn_ds_vals = [results[s]["dn_ds_ratio"] for s in species_names_d]
x_d = np.arange(len(species_names_d))

abc_max = max(
    max(syn_vals) if syn_vals else 0,
    max(non_vals) if non_vals else 0,
    max(stop_vals) if stop_vals else 0
)
ymax = abc_max * 1.08 if abc_max > 0 else 1.0

finite_dn_ds_vals = [v for v in dn_ds_vals if np.isfinite(v)]
dn_ds_ymax = max(finite_dn_ds_vals) * 1.08 if finite_dn_ds_vals else 1.0

# Plots
fig, axes = plt.subplots(4, 1, figsize=(18, 18), sharex=False)

axes[0].bar(x_abc, syn_vals, color=bar_colors, edgecolor="black", linewidth=0.6, zorder=3)
axes[0].set_title("Synonymous")
axes[0].set_ylabel("Proportion of mutations")
axes[0].grid(axis="y", alpha=0.25, linewidth=0.6)
axes[0].set_axisbelow(True)
axes[0].set_ylim(0, ymax)
axes[0].set_xticks(x_abc)
axes[0].set_xticklabels([])
axes[0].text(-0.08, 1.02, "a)", transform=axes[0].transAxes, fontsize=15, fontweight="bold", va="bottom")

axes[1].bar(x_abc, non_vals, color=bar_colors, edgecolor="black", linewidth=0.6, zorder=3)
axes[1].set_title("Nonsynonymous")
axes[1].set_ylabel("Proportion of mutations")
axes[1].grid(axis="y", alpha=0.25, linewidth=0.6)
axes[1].set_axisbelow(True)
axes[1].set_ylim(0, ymax)
axes[1].set_xticks(x_abc)
axes[1].set_xticklabels([])
axes[1].text(-0.08, 1.02, "b)", transform=axes[1].transAxes, fontsize=15, fontweight="bold", va="bottom")

axes[2].bar(x_abc, stop_vals, color=bar_colors, edgecolor="black", linewidth=0.6, zorder=3)
axes[2].set_title("Nonsense")
axes[2].set_ylabel("Proportion of mutations")
axes[2].grid(axis="y", alpha=0.25, linewidth=0.6)
axes[2].set_axisbelow(True)
axes[2].set_ylim(0, ymax)
axes[2].set_xticks(x_abc)
apply_species_xticklabels(axes[2], display_names, rotation=60, fontsize=9)
axes[2].text(-0.08, 1.02, "c)", transform=axes[2].transAxes, fontsize=15, fontweight="bold", va="bottom")

axes[3].bar(x_d, dn_ds_vals, color=bar_colors_d, edgecolor="black", linewidth=0.6, zorder=3)
axes[3].set_title("dN/dS ratio")
axes[3].set_ylabel("dN/dS")
axes[3].grid(axis="y", alpha=0.25, linewidth=0.6)
axes[3].set_axisbelow(True)
axes[3].set_ylim(0, dn_ds_ymax)
axes[3].set_xticks(x_d)
apply_species_xticklabels(axes[3], display_names_d, rotation=60, fontsize=9)
axes[3].text(-0.08, 1.02, "d)", transform=axes[3].transAxes, fontsize=15, fontweight="bold", va="bottom")

fig.suptitle("Coding-consequence mutation proportions across species", fontsize=16, y=0.98)

# Final layout
fig.subplots_adjust(left=0.08, right=0.98, top=0.91, bottom=0.08, hspace=0.55)

desired_ab_gap = 0.05
desired_bc_gap = 0.05
desired_cd_gap = 0.13

pos_a = axes[0].get_position()
pos_b = axes[1].get_position()
pos_c = axes[2].get_position()
pos_d = axes[3].get_position()

h_b = pos_b.height
h_c = pos_c.height
h_d = pos_d.height

new_b_top = pos_a.y0 - desired_ab_gap
new_b_y0 = new_b_top - h_b
axes[1].set_position([pos_b.x0, new_b_y0, pos_b.width, h_b])

pos_b = axes[1].get_position()

new_c_top = pos_b.y0 - desired_bc_gap
new_c_y0 = new_c_top - h_c
axes[2].set_position([pos_c.x0, new_c_y0, pos_c.width, h_c])

pos_c = axes[2].get_position()

new_d_top = pos_c.y0 - desired_cd_gap
new_d_y0 = new_d_top - h_d
axes[3].set_position([pos_d.x0, new_d_y0, pos_d.width, h_d])

plt.show()
