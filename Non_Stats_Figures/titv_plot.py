import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

# Settings
VCF_DIR = Path("/shared/team/2025-masters-project/people/justin/all_genomes_singletonsandwindowmasked")
REF_DIR = Path("/shared/team/2025-masters-project/people/justin/all_genomes_referenceandFFA_files")

ALLOW_FILTER = {"PASS", "."}
SNP_ONLY_BIALLELIC = True
ENFORCE_K1_AFTER_MASKING = True
SKIP_ALL_REF_ROWS = False

VCF_SUFFIX = "_biallelic_k1_singletons_windowmasked.vcf"
REF_SUFFIX = ".fna"

BASES = {"A", "C", "G", "T"}

TRANSITIONS = {("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")}
TRANSVERSIONS = {
    ("A", "C"), ("A", "T"),
    ("C", "A"), ("C", "G"),
    ("G", "C"), ("G", "T"),
    ("T", "A"), ("T", "G"),
}

ALL_CLASSES = [
    ("A", "C"), ("A", "G"), ("A", "T"),
    ("C", "A"), ("C", "G"), ("C", "T"),
    ("G", "A"), ("G", "C"), ("G", "T"),
    ("T", "A"), ("T", "C"), ("T", "G"),
]

# Species label formatting
def format_species_label(label):
    label = str(label).strip().replace("_", " ")
    parts = label.split()
    if not parts:
        return ""
    return " ".join([parts[0].capitalize()] + [p.lower() for p in parts[1:]])

def apply_species_xticklabels(ax, labels, rotation=60, fontsize=9):
    formatted = [format_species_label(lbl) for lbl in labels]
    ax.set_xticklabels(formatted, rotation=rotation, ha="right", rotation_mode="anchor", fontsize=fontsize)
    for tick in ax.get_xticklabels():
        tick.set_fontstyle("italic")

def ansi_italic(text):
    return f"\x1b[3m{text}\x1b[0m"

# FASTA base composition
def read_fasta_base_counts(fasta_path: str) -> Counter:
    counts = Counter()
    with open(fasta_path) as f:
        for line in f:
            if line.startswith(">"):
                continue
            for b in line.strip().upper():
                if b in BASES:
                    counts[b] += 1
    return counts

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
def analyse_species_titv(vcf_path: Path, ref_fasta_path: Path):
    counts = Counter()
    total_rows = 0
    used_rows = 0
    skipped_all_ref = 0
    skipped_not_k1 = 0

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
            ref = col[3].upper()
            alt = col[4].upper()
            if (ref, alt) in ALL_CLASSES:
                counts[(ref, alt)] += 1
                used_rows += 1

    raw_ti = sum(counts[c] for c in TRANSITIONS)
    raw_tv = sum(counts[c] for c in TRANSVERSIONS)
    raw_ratio = raw_ti / raw_tv if raw_tv > 0 else float("inf")
    raw_total = raw_ti + raw_tv
    raw_ti_prop = raw_ti / raw_total if raw_total > 0 else 0.0
    raw_tv_prop = raw_tv / raw_total if raw_total > 0 else 0.0

    ref_counts = read_fasta_base_counts(ref_fasta_path)

    norm_class_rates = {}
    for ref, alt in ALL_CLASSES:
        opp = ref_counts[ref]
        norm_class_rates[(ref, alt)] = (counts[(ref, alt)] / opp) if opp > 0 else 0.0

    norm_ti = sum(norm_class_rates[c] for c in TRANSITIONS)
    norm_tv = sum(norm_class_rates[c] for c in TRANSVERSIONS)
    norm_ratio = norm_ti / norm_tv if norm_tv > 0 else float("inf")
    norm_total = norm_ti + norm_tv
    norm_ti_prop = norm_ti / norm_total if norm_total > 0 else 0.0
    norm_tv_prop = norm_tv / norm_total if norm_total > 0 else 0.0

    return {
        "total_rows": total_rows,
        "used_rows": used_rows,
        "skipped_all_ref": skipped_all_ref,
        "skipped_not_k1": skipped_not_k1,
        "raw_ti": raw_ti,
        "raw_tv": raw_tv,
        "raw_titv": raw_ratio,
        "raw_ti_prop": raw_ti_prop,
        "raw_tv_prop": raw_tv_prop,
        "norm_ti": norm_ti,
        "norm_tv": norm_tv,
        "norm_titv": norm_ratio,
        "norm_ti_prop": norm_ti_prop,
        "norm_tv_prop": norm_tv_prop,
    }

# Run all species
vcf_files = sorted(VCF_DIR.glob(f"*{VCF_SUFFIX}"))
results = {}

for vcf_path in vcf_files:
    species = species_name_from_vcf(vcf_path)
    ref_fasta_path = REF_DIR / f"{species}{REF_SUFFIX}"

    if not ref_fasta_path.exists():
        print(f"Skipping {species}: missing reference FASTA {ref_fasta_path.name}")
        continue

    res = analyse_species_titv(vcf_path, ref_fasta_path)
    results[species] = res

    raw_str = f"{res['raw_titv']:.4f}" if np.isfinite(res["raw_titv"]) else "inf"
    norm_str = f"{res['norm_titv']:.4f}" if np.isfinite(res["norm_titv"]) else "inf"

    print(f"Processed {species}")
    print(f"  raw Ti/Tv  = {raw_str}")
    print(f"  norm Ti/Tv = {norm_str}")

if len(results) == 0:
    raise ValueError("No species were processed. Check VCF_DIR, REF_DIR, and filename matching.")

species_names = sorted(results.keys())
display_names = [format_species_label(s) for s in species_names]

# Summary table
print("\nTi/Tv summary table:\n")

header = (
    f"{'Species':30}"
    f"{'Raw_Ti':>10}"
    f"{'Raw_Tv':>10}"
    f"{'Raw_TiTv':>12}"
    f"{'Norm_Ti_rate':>16}"
    f"{'Norm_Tv_rate':>16}"
    f"{'Norm_TiTv':>12}"
    f"{'Norm_Ti_prop':>14}"
    f"{'Norm_Tv_prop':>14}"
)

print(header)
print("-" * len(header))

def fmt(x, sci=False):
    if not np.isfinite(x):
        return "inf"
    return f"{x:.6e}" if sci else f"{x:.4f}"

for s in species_names:
    r = results[s]
    species_for_table = ansi_italic(format_species_label(s))
    print(
        f"{species_for_table:30}"
        f"{r['raw_ti']:10d}"
        f"{r['raw_tv']:10d}"
        f"{fmt(r['raw_titv']):>12}"
        f"{fmt(r['norm_ti'], sci=True):>16}"
        f"{fmt(r['norm_tv'], sci=True):>16}"
        f"{fmt(r['norm_titv']):>12}"
        f"{fmt(r['norm_ti_prop']):>14}"
        f"{fmt(r['norm_tv_prop']):>14}"
    )

# Colours
master_species_order = sorted(results.keys())
cmap = plt.colormaps["plasma"]
n_species = len(master_species_order)

species_colors = {
    species: cmap(0.35 + 0.5 * (i / (n_species - 1 if n_species > 1 else 1)))
    for i, species in enumerate(master_species_order)
}

# Order for plotting
species_order = sorted(results.keys(), key=lambda s: results[s]["norm_ti_prop"], reverse=True)
display_names_ordered = [format_species_label(s) for s in species_order]
bar_colors = [species_colors[s] for s in species_order]

transition_vals = [results[s]["norm_ti_prop"] for s in species_order]
transversion_vals = [results[s]["norm_tv_prop"] for s in species_order]
titv_vals = [results[s]["norm_titv"] for s in species_order]

prop_max = max(max(transition_vals) if transition_vals else 0, max(transversion_vals) if transversion_vals else 0)
prop_ymax = prop_max * 1.15 if prop_max > 0 else 1.0

finite_titv = [v for v in titv_vals if np.isfinite(v)]
titv_ymax = max(finite_titv) * 1.15 if finite_titv else 1.0

# Plot
fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
x = np.arange(len(species_order))

axes[0].bar(x, transition_vals, color=bar_colors, edgecolor="black", linewidth=0.6, zorder=3)
axes[0].set_title("Transitions")
axes[0].set_ylabel("Proportion of mutations")
axes[0].grid(axis="y", alpha=0.25, linewidth=0.6)
axes[0].set_axisbelow(True)
axes[0].set_ylim(0, prop_ymax)

axes[1].bar(x, transversion_vals, color=bar_colors, edgecolor="black", linewidth=0.6, zorder=3)
axes[1].set_title("Transversions")
axes[1].set_ylabel("Proportion of mutations")
axes[1].grid(axis="y", alpha=0.25, linewidth=0.6)
axes[1].set_axisbelow(True)
axes[1].set_ylim(0, prop_ymax)

axes[2].bar(x, titv_vals, color=bar_colors, edgecolor="black", linewidth=0.6, zorder=3)
axes[2].set_title("Ti/Tv ratio")
axes[2].set_ylabel("Ti/Tv ratio")
axes[2].grid(axis="y", alpha=0.25, linewidth=0.6)
axes[2].set_axisbelow(True)
axes[2].set_ylim(0, titv_ymax)
axes[2].set_xticks(x)
apply_species_xticklabels(axes[2], display_names_ordered, rotation=60, fontsize=9)
axes[2].axhline(1.0, color="black", linewidth=1.2, linestyle=":")

panel_labels = ["a)", "b)", "c)"]
for ax, label in zip(axes, panel_labels):
    ax.text(-0.08, 1.05, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=14, fontweight="bold")

fig.suptitle("Transition and transversion patterns across species", fontsize=16, y=0.99)
plt.tight_layout(rect=[0.05, 0, 1, 0.97])
plt.show()
