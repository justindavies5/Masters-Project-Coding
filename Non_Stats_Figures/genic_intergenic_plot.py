import matplotlib.pyplot as plt
from pathlib import Path
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
def analyse_species_genic_vs_intergenic(vcf_path: Path, fasta_path: Path, gff_path: Path):
    ref_seqs = read_fasta_dict(fasta_path)
    features = parse_gff_features(gff_path, feature_types=FEATURE_TYPES)
    genic_mask, genic_bases, intergenic_bases, total_bases = build_genic_mask(ref_seqs, features)

    in_genic = 0
    intergenic = 0
    total_rows = 0
    used_rows = 0
    skipped_all_ref = 0
    skipped_not_k1 = 0
    skipped_missing_seq = 0
    skipped_bad_pos = 0

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
            if genic_mask[chrom][pos] == 1:
                in_genic += 1
            else:
                intergenic += 1
            used_rows += 1

    raw_ratio = in_genic / intergenic if intergenic > 0 else float("inf")
    raw_total = in_genic + intergenic
    raw_genic_prop = in_genic / raw_total if raw_total > 0 else 0.0
    raw_intergenic_prop = intergenic / raw_total if raw_total > 0 else 0.0
    norm_genic = in_genic / genic_bases if genic_bases > 0 else 0.0
    norm_intergenic = intergenic / intergenic_bases if intergenic_bases > 0 else 0.0
    norm_ratio = norm_genic / norm_intergenic if norm_intergenic > 0 else float("inf")
    norm_total = norm_genic + norm_intergenic
    norm_genic_prop = norm_genic / norm_total if norm_total > 0 else 0.0
    norm_intergenic_prop = norm_intergenic / norm_total if norm_total > 0 else 0.0

    return {
        "total_rows": total_rows,
        "used_rows": used_rows,
        "skipped_all_ref": skipped_all_ref,
        "skipped_not_k1": skipped_not_k1,
        "skipped_missing_seq": skipped_missing_seq,
        "skipped_bad_pos": skipped_bad_pos,
        "genic_bases": genic_bases,
        "intergenic_bases": intergenic_bases,
        "total_bases": total_bases,
        "raw_genic": in_genic,
        "raw_intergenic": intergenic,
        "raw_ratio": raw_ratio,
        "raw_genic_prop": raw_genic_prop,
        "raw_intergenic_prop": raw_intergenic_prop,
        "norm_genic": norm_genic,
        "norm_intergenic": norm_intergenic,
        "norm_ratio": norm_ratio,
        "norm_genic_prop": norm_genic_prop,
        "norm_intergenic_prop": norm_intergenic_prop,
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

    res = analyse_species_genic_vs_intergenic(vcf_path, fasta_path, gff_path)
    results[species] = res

    raw_ratio_str = f"{res['raw_ratio']:.4f}" if np.isfinite(res["raw_ratio"]) else "inf"
    norm_ratio_str = f"{res['norm_ratio']:.4f}" if np.isfinite(res["norm_ratio"]) else "inf"

    print(f"Processed {species}")
    print(f"  raw genic/intergenic ratio  = {raw_ratio_str}")
    print(f"  norm genic/intergenic ratio = {norm_ratio_str}")

if len(results) == 0:
    raise ValueError("No species were processed. Check paths and filename matching.")

species_names = sorted(results.keys())

# Table
SPECIES_WIDTH = 25

print("\nGenic vs intergenic summary table:\n")

header = (
    f"{'Species':<{SPECIES_WIDTH}}"
    f"{'Raw_genic':>12}"
    f"{'Raw_intergenic':>16}"
    f"{'Raw_ratio':>12}"
    f"{'Genic_bases':>14}"
    f"{'Intergenic_bases':>18}"
    f"{'Norm_genic':>14}"
    f"{'Norm_intergenic':>18}"
    f"{'Norm_ratio':>12}"
)
print(header)
print("-" * len(header))

for s in species_names:
    r = results[s]
    raw_ratio_str = f"{r['raw_ratio']:.4f}" if np.isfinite(r["raw_ratio"]) else "inf"
    norm_ratio_str = f"{r['norm_ratio']:.4f}" if np.isfinite(r["norm_ratio"]) else "inf"
    species_label = format_species_label(s)
    print(
        f"{species_label:<{SPECIES_WIDTH}}"
        f"{r['raw_genic']:12d}"
        f"{r['raw_intergenic']:16d}"
        f"{raw_ratio_str:>12}"
        f"{r['genic_bases']:14d}"
        f"{r['intergenic_bases']:18d}"
        f"{r['norm_genic']:14.6e}"
        f"{r['norm_intergenic']:18.6e}"
        f"{norm_ratio_str:>12}"
    )

# Colours
master_species_order = sorted(results.keys())
cmap = plt.colormaps[COLORMAP_NAME]
n_species = len(master_species_order)

species_colors = {
    species: cmap(0.35 + 0.5 * (i / (n_species - 1 if n_species > 1 else 1)))
    for i, species in enumerate(master_species_order)
}

# Values
species_order = sorted(
    results.keys(),
    key=lambda s: results[s]["norm_genic_prop"],
    reverse=True
)

display_names_ordered = [format_species_label(s) for s in species_order]
bar_colors = [species_colors[s] for s in species_order]

genic_vals = [results[s]["norm_genic_prop"] for s in species_order]
intergenic_vals = [results[s]["norm_intergenic_prop"] for s in species_order]
ratio_vals = [results[s]["norm_ratio"] for s in species_order]

prop_max = max(
    max(genic_vals) if genic_vals else 0,
    max(intergenic_vals) if intergenic_vals else 0
)
prop_ymax = prop_max * 1.15 if prop_max > 0 else 1.0

finite_ratio = [v for v in ratio_vals if np.isfinite(v)]
ratio_ymax = max(finite_ratio) * 1.15 if finite_ratio else 1.0

# Plot
fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
x = np.arange(len(species_order))

axes[0].bar(x, genic_vals, color=bar_colors, edgecolor="black", linewidth=0.6, zorder=3)
axes[0].set_title("Genic")
axes[0].set_ylabel("Proportion of mutations")
axes[0].grid(axis="y", alpha=0.25, linewidth=0.6)
axes[0].set_axisbelow(True)
axes[0].set_ylim(0, prop_ymax)

axes[1].bar(x, intergenic_vals, color=bar_colors, edgecolor="black", linewidth=0.6, zorder=3)
axes[1].set_title("Intergenic")
axes[1].set_ylabel("Proportion of mutations")
axes[1].grid(axis="y", alpha=0.25, linewidth=0.6)
axes[1].set_axisbelow(True)
axes[1].set_ylim(0, prop_ymax)

axes[2].bar(x, ratio_vals, color=bar_colors, edgecolor="black", linewidth=0.6, zorder=3)
axes[2].set_title("Genic: intergenic ratio")
axes[2].set_ylabel("Genic/intergenic ratio")
axes[2].grid(axis="y", alpha=0.25, linewidth=0.6)
axes[2].set_axisbelow(True)
axes[2].set_ylim(0, ratio_ymax)
axes[2].set_xticks(x)
apply_species_xticklabels(axes[2], display_names_ordered, rotation=60, fontsize=9)
axes[2].axhline(1.0, color="black", linewidth=1.2, linestyle=":")

panel_labels = ["a)", "b)", "c)"]
for ax, label in zip(axes, panel_labels):
    ax.text(-0.08, 1.05, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=14, fontweight="bold")

fig.suptitle("Genic vs intergenic mutation proportions across species", fontsize=16, y=0.99)
plt.tight_layout(rect=[0.05, 0, 1, 0.97])
plt.show()
