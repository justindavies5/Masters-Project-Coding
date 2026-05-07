import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd

# Settings
SINGLETON_DIR = Path("/shared/team/2025-masters-project/people/justin/all_genomes_singletons")
WINDOWMASKED_DIR = Path("/shared/team/2025-masters-project/people/justin/all_genomes_singletonsandwindowmasked")
ANNOT_DIR = Path("/shared/team/2025-masters-project/people/justin/all_genomes_referenceandFFA_files")

SINGLETON_SUFFIX = "_biallelic_k1_singletons.vcf"
WINDOWMASKED_SUFFIX = "_biallelic_k1_singletons_windowmasked.vcf"
FASTA_SUFFIX = ".fna"
GFF_SUFFIX = ".gff"

ALLOW_FILTER = {"PASS", "."}
SNP_ONLY_BIALLELIC = True
MISSING_POSTMASK_AS_REMOVED = True
FEATURE_TYPES = {"CDS"}

COLORMAP_NAME = "plasma"

BASES = {"A", "C", "G", "T"}

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

# GFF / genic mask
def parse_gff_features(gff_path: str, feature_types={"CDS"}):
    features = []
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
            if feature_type not in feature_types:
                continue
            try:
                start = int(start)
                end = int(end)
            except ValueError:
                continue
            if start > end:
                start, end = end, start
            features.append((seqid, start, end))
    return features

def build_genic_mask(ref_seqs, features):
    genic_mask = {}
    for seqid, seq in ref_seqs.items():
        genic_mask[seqid] = np.zeros(len(seq) + 1, dtype=np.int8)

    for seqid, start, end in features:
        if seqid not in genic_mask:
            continue
        n = len(ref_seqs[seqid])
        start = max(1, start)
        end = min(n, end)
        if start > end:
            continue
        genic_mask[seqid][start:end+1] = 1

    genic_bases = 0
    intergenic_bases = 0
    total_bases = 0

    for seqid, seq in ref_seqs.items():
        mask = genic_mask[seqid]
        for pos in range(1, len(seq) + 1):
            if seq[pos - 1] not in BASES:
                continue
            total_bases += 1
            if mask[pos] == 1:
                genic_bases += 1
            else:
                intergenic_bases += 1

    return genic_mask, genic_bases, intergenic_bases, total_bases

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

def count_alt_carriers(sample_fields, gt_index: int) -> int:
    c = 0
    for sf in sample_fields:
        if is_alt_carrier(sf, gt_index):
            c += 1
    return c

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

def species_name_from_singleton_vcf(vcf_path: Path) -> str:
    if vcf_path.name.endswith(SINGLETON_SUFFIX):
        return vcf_path.name[:-len(SINGLETON_SUFFIX)]
    return vcf_path.stem

def make_variant_key(col):
    return (col[0], col[1], col[3].upper(), col[4].upper())

def read_postmask_carrier_counts(vcf_path: Path):
    carrier_dict = {}
    with open(vcf_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            col = line.rstrip("\n").split("\t")
            if not is_pass_snp_row(col):
                continue
            gt_idx = gt_index_from_format(col[8])
            if gt_idx is None:
                continue
            carriers = count_alt_carriers(col[9:], gt_idx)
            carrier_dict[make_variant_key(col)] = carriers
    return carrier_dict

# Analyse one species
def analyse_removed_recombination(singleton_vcf: Path, windowmasked_vcf: Path, fasta_path: Path, gff_path: Path):
    ref_seqs = read_fasta_dict(fasta_path)
    features = parse_gff_features(gff_path, feature_types=FEATURE_TYPES)
    genic_mask, genic_bases, intergenic_bases, total_bases = build_genic_mask(ref_seqs, features)

    postmask_counts = read_postmask_carrier_counts(windowmasked_vcf)

    removed_total = 0
    removed_genic = 0
    removed_intergenic = 0
    total_singleton_rows = 0
    removed_missing_post = 0
    skipped_bad_pos = 0
    skipped_missing_seq = 0

    with open(singleton_vcf, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            col = line.rstrip("\n").split("\t")
            if not is_pass_snp_row(col):
                continue
            gt_idx = gt_index_from_format(col[8])
            if gt_idx is None:
                continue
            pre_carriers = count_alt_carriers(col[9:], gt_idx)
            if pre_carriers != 1:
                continue
            total_singleton_rows += 1
            key = make_variant_key(col)

            if key not in postmask_counts:
                if MISSING_POSTMASK_AS_REMOVED:
                    post_carriers = 0
                    removed_missing_post += 1
                else:
                    removed_missing_post += 1
                    continue
            else:
                post_carriers = postmask_counts[key]

            if post_carriers == 0:
                chrom = col[0]
                if chrom not in ref_seqs or chrom not in genic_mask:
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
                ref_base = ref_seqs[chrom][pos - 1].upper()
                if ref_base not in BASES:
                    continue
                removed_total += 1
                if genic_mask[chrom][pos] == 1:
                    removed_genic += 1
                else:
                    removed_intergenic += 1

    raw_ratio = removed_genic / removed_intergenic if removed_intergenic > 0 else float("inf")
    norm_total_removed = removed_total / total_bases if total_bases > 0 else 0.0
    norm_genic = removed_genic / genic_bases if genic_bases > 0 else 0.0
    norm_intergenic = removed_intergenic / intergenic_bases if intergenic_bases > 0 else 0.0
    norm_ratio = norm_genic / norm_intergenic if norm_intergenic > 0 else float("inf")
    norm_sum = norm_genic + norm_intergenic
    norm_genic_prop = norm_genic / norm_sum if norm_sum > 0 else 0.0
    norm_intergenic_prop = norm_intergenic / norm_sum if norm_sum > 0 else 0.0

    return {
        "total_singleton_rows": total_singleton_rows,
        "removed_total": removed_total,
        "removed_genic": removed_genic,
        "removed_intergenic": removed_intergenic,
        "raw_ratio": raw_ratio,
        "genic_bases": genic_bases,
        "intergenic_bases": intergenic_bases,
        "total_bases": total_bases,
        "norm_total_removed": norm_total_removed,
        "norm_genic": norm_genic,
        "norm_intergenic": norm_intergenic,
        "norm_ratio": norm_ratio,
        "norm_genic_prop": norm_genic_prop,
        "norm_intergenic_prop": norm_intergenic_prop,
        "removed_missing_post": removed_missing_post,
        "skipped_bad_pos": skipped_bad_pos,
        "skipped_missing_seq": skipped_missing_seq,
    }

# Run all species
singleton_files = sorted(SINGLETON_DIR.glob(f"*{SINGLETON_SUFFIX}"))
results = {}

for singleton_vcf in singleton_files:
    species = species_name_from_singleton_vcf(singleton_vcf)
    windowmasked_vcf = WINDOWMASKED_DIR / f"{species}{WINDOWMASKED_SUFFIX}"
    fasta_path = ANNOT_DIR / f"{species}{FASTA_SUFFIX}"
    gff_path = ANNOT_DIR / f"{species}{GFF_SUFFIX}"

    if not windowmasked_vcf.exists():
        print(f"Skipping {species}: missing windowmasked VCF")
        continue
    if not fasta_path.exists():
        print(f"Skipping {species}: missing FASTA")
        continue
    if not gff_path.exists():
        print(f"Skipping {species}: missing GFF")
        continue

    res = analyse_removed_recombination(singleton_vcf, windowmasked_vcf, fasta_path, gff_path)
    results[species] = res

    print(f"Processed {species}")
    print(f"  removed total        = {res['removed_total']}")
    print(f"  removed genic        = {res['removed_genic']}")
    print(f"  removed intergenic   = {res['removed_intergenic']}")
    print(f"  norm total removed   = {res['norm_total_removed']:.6e}")
    print(f"  norm genic removed   = {res['norm_genic']:.6e}")
    print(f"  norm intergenic rem. = {res['norm_intergenic']:.6e}")

if len(results) == 0:
    raise ValueError("No species were processed. Check paths and filename matching.")

species_names = sorted(results.keys())

# Summary table
print("\nRecombination-removed summary table:\n")

def format_species_label(species_name: str) -> str:
    label = species_name.replace("_", " ").strip()
    if not label:
        return label
    return label[0].upper() + label[1:].lower()

def apply_species_xticklabels(ax, species_names, rotation=60, fontsize=10):
    labels = [format_species_label(s) for s in species_names]
    ax.set_xticks(np.arange(len(species_names)))
    ax.set_xticklabels(labels, rotation=rotation, ha="right", fontsize=fontsize)
    for tick in ax.get_xticklabels():
        tick.set_fontstyle("italic")

SPECIES_WIDTH = max(len(format_species_label(s)) for s in species_names) + 2

header = (
    f"{'Species':<{SPECIES_WIDTH}}"
    f"{'Removed_total':>14}"
    f"{'Removed_genic':>14}"
    f"{'Removed_inter':>14}"
    f"{'Genic_bases':>12}"
    f"{'Intergenic_bases':>18}"
    f"{'Total_bases':>12}"
    f"{'Norm_total':>14}"
    f"{'Norm_genic':>14}"
    f"{'Norm_inter':>14}"
    f"{'Norm_ratio':>12}"
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
        f"{r['removed_total']:14d}"
        f"{r['removed_genic']:14d}"
        f"{r['removed_intergenic']:14d}"
        f"{r['genic_bases']:12d}"
        f"{r['intergenic_bases']:18d}"
        f"{r['total_bases']:12d}"
        f"{r['norm_total_removed']:14.6e}"
        f"{r['norm_genic']:14.6e}"
        f"{r['norm_intergenic']:14.6e}"
        f"{fmt_ratio(r['norm_ratio']):>12}"
    )

# Colours
master_species_order = sorted(results.keys())
cmap = plt.colormaps[COLORMAP_NAME]
n_species = len(master_species_order)

species_colors = {
    species: cmap(0.35 + 0.5 * (i / (n_species - 1 if n_species > 1 else 1)))
    for i, species in enumerate(master_species_order)
}

# Panel a order
species_order_total = sorted(results.keys(), key=lambda s: results[s]["norm_total_removed"], reverse=True)
display_names_total = [s.replace("_", " ") for s in species_order_total]
colors_total = [species_colors[s] for s in species_order_total]
vals_total = [results[s]["norm_total_removed"] * 100 for s in species_order_total]

# Panels b/c/d order
species_order = sorted(results.keys(), key=lambda s: results[s]["norm_genic_prop"], reverse=True)
display_names = [s.replace("_", " ") for s in species_order]
colors = [species_colors[s] for s in species_order]

vals_genic = [results[s]["norm_genic_prop"] * 100 for s in species_order]
vals_inter = [results[s]["norm_intergenic_prop"] * 100 for s in species_order]
vals_ratio = [results[s]["norm_ratio"] for s in species_order]

# Y limits
a_max = max(vals_total) if vals_total else 0
a_ymax = a_max * 1.15 if a_max > 0 else 1.0

bc_max = max(max(vals_genic) if vals_genic else 0, max(vals_inter) if vals_inter else 0)
bc_ymax = bc_max * 1.15 if bc_max > 0 else 1.0

finite_ratio = [v for v in vals_ratio if np.isfinite(v)]
d_ymax = max(finite_ratio) * 1.15 if finite_ratio else 1.0

# Plot
fig, axes = plt.subplots(4, 1, figsize=(18, 18), sharex=False)

x0 = np.arange(len(species_order_total))
axes[0].bar(x0, vals_total, color=colors_total, edgecolor="black", linewidth=0.6, zorder=3)
axes[0].set_title("Total proportion of nucleotides removed due to recombination")
axes[0].set_ylabel("Percentage removed (%)")
apply_species_xticklabels(axes[0], species_order_total, rotation=60, fontsize=10)
axes[0].grid(axis="y", alpha=0.25, linewidth=0.6)
axes[0].set_axisbelow(True)
axes[0].set_ylim(0, a_ymax)

x = np.arange(len(species_order))

axes[1].bar(x, vals_genic, color=colors, edgecolor="black", linewidth=0.6, zorder=3)
axes[1].set_title("Genic proportion of recombination-removed nucleotides")
axes[1].set_ylabel("Percentage removed (%)")
axes[1].set_xticks(x)
axes[1].set_xticklabels([])
axes[1].grid(axis="y", alpha=0.25, linewidth=0.6)
axes[1].set_axisbelow(True)
axes[1].set_ylim(0, bc_ymax)

axes[2].bar(x, vals_inter, color=colors, edgecolor="black", linewidth=0.6, zorder=3)
axes[2].set_title("Intergenic proportion of recombination-removed nucleotides")
axes[2].set_ylabel("Percentage removed (%)")
axes[2].set_xticks(x)
axes[2].set_xticklabels([])
axes[2].grid(axis="y", alpha=0.25, linewidth=0.6)
axes[2].set_axisbelow(True)
axes[2].set_ylim(0, bc_ymax)

axes[3].bar(x, vals_ratio, color=colors, edgecolor="black", linewidth=0.6, zorder=3)
axes[3].set_title("Genic: intergenic ratio")
axes[3].set_ylabel("Genic/intergenic ratio")
apply_species_xticklabels(axes[3], species_order, rotation=60, fontsize=10)
axes[3].grid(axis="y", alpha=0.25, linewidth=0.6)
axes[3].set_axisbelow(True)
axes[3].set_ylim(0, d_ymax)
axes[3].axhline(1.0, color="black", linewidth=1.2, linestyle=":")

for ax, label in zip(axes, ["a)", "b)", "c)", "d)"]):
    ax.text(-0.08, 1.05, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=14, fontweight="bold")

fig.suptitle("Proportion of nucleotides removed due to recombination across species", fontsize=16, y=0.99)

fig.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.08, hspace=1.25)

pos_b = axes[1].get_position()
pos_c = axes[2].get_position()
pos_d = axes[3].get_position()

desired_bc_gap = 0.035
desired_cd_gap = 0.035

new_c_top = pos_b.y0 - desired_bc_gap
new_c_y0 = new_c_top - pos_c.height
axes[2].set_position([pos_c.x0, new_c_y0, pos_c.width, pos_c.height])

pos_c_new = axes[2].get_position()

new_d_top = pos_c_new.y0 - desired_cd_gap
new_d_y0 = new_d_top - pos_d.height
axes[3].set_position([pos_d.x0, new_d_y0, pos_d.width, pos_d.height])

plt.show()
