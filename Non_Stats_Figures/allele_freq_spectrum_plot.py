import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from collections import Counter
from pathlib import Path

# Settings
VCF_DIR = Path("/shared/team/2025-masters-project/people/justin/all_genomes_windowmasked")
REF_DIR = Path("/shared/team/2025-masters-project/people/justin/all_genomes_referenceandFFA_files")

VCF_SUFFIX = "_parsnp_windowmasked.vcf"
REF_SUFFIX = ".fna"

KS = [1, 2, 3]
ALLOW_FILTER = {"PASS", "."}

CLASSES = ["C>A", "C>G", "C>T", "T>A", "T>G", "T>C"]
PANEL_ORDER = ["C>A", "T>A", "C>G", "T>G", "C>T", "T>C"]

BASES = {"A", "C", "G", "T"}
COMP = {"A": "T", "T": "A", "C": "G", "G": "C"}

# Label helpers
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

# Helpers
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

def is_biallelic_snp_pass(col) -> bool:
    if col[6] not in ALLOW_FILTER:
        return False
    ref = col[3].upper()
    alt = col[4].upper()
    if "," in alt:
        return False
    if ref not in BASES or alt not in BASES:
        return False
    if ref == alt:
        return False
    return True

def to_6class(ref: str, alt: str) -> str:
    ref = ref.upper()
    alt = alt.upper()
    if ref in {"A", "G"}:
        ref = COMP[ref]
        alt = COMP[alt]
    return f"{ref}>{alt}"

def safe_norm(vec):
    s = sum(vec.values())
    return {k: (v / s if s > 0 else 0.0) for k, v in vec.items()}

def species_name_from_vcf(vcf_path: Path) -> str:
    if vcf_path.name.endswith(VCF_SUFFIX):
        return vcf_path.name[:-len(VCF_SUFFIX)]
    return vcf_path.stem

def lighten_color(color, amount=0.5):
    c = np.array(mcolors.to_rgb(color))
    white = np.array([1.0, 1.0, 1.0])
    return tuple((1 - amount) * c + amount * white)

def make_k_shades(base_color):
    amounts = {1: 0.55, 2: 0.28, 3: 0.00}
    return {k: lighten_color(base_color, amounts[k]) for k in KS}

# Process one species
def get_k_spectra(vcf_path: Path, ref_fasta_path: Path):
    directional_counts_by_k = {k: Counter() for k in KS}
    totals_by_k = {k: 0 for k in KS}

    n_samples = None
    total_rows = 0
    used_rows = 0

    with open(vcf_path, "r") as f:
        for line in f:
            if line.startswith("#CHROM"):
                header = line.rstrip("\n").split("\t")
                n_samples = len(header) - 9
                break

    if n_samples is None:
        raise ValueError(f"Could not find #CHROM header line in {vcf_path}")

    with open(vcf_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            total_rows += 1
            col = line.rstrip("\n").split("\t")
            if not is_biallelic_snp_pass(col):
                continue
            gt_idx = gt_index_from_format(col[8])
            if gt_idx is None:
                continue
            k = 0
            for sample_field in col[9:]:
                if is_alt_carrier(sample_field, gt_idx):
                    k += 1
                    if k > max(KS):
                        break
            if k not in totals_by_k:
                continue
            ref = col[3].upper()
            alt = col[4].upper()
            directional_counts_by_k[k][(ref, alt)] += 1
            totals_by_k[k] += 1
            used_rows += 1

    ref_counts = read_fasta_base_counts(ref_fasta_path)
    class_props_by_k = {k: {cls: 0.0 for cls in CLASSES} for k in KS}

    for k in KS:
        directional_rates = {}
        for (ref, alt), count in directional_counts_by_k[k].items():
            opp = ref_counts[ref]
            directional_rates[(ref, alt)] = count / opp if opp > 0 else 0.0
        collapsed_rates = {cls: 0.0 for cls in CLASSES}
        for (ref, alt), rate in directional_rates.items():
            cls = to_6class(ref, alt)
            collapsed_rates[cls] += rate
        collapsed_props = safe_norm(collapsed_rates)
        class_props_by_k[k] = collapsed_props

    return class_props_by_k, totals_by_k, n_samples, total_rows, used_rows

# Run all species
vcf_files = sorted(VCF_DIR.glob(f"*{VCF_SUFFIX}"))

species_results = {}
species_site_counts = {}

for vcf_path in vcf_files:
    species = species_name_from_vcf(vcf_path)
    ref_fasta_path = REF_DIR / f"{species}{REF_SUFFIX}"

    if not ref_fasta_path.exists():
        print(f"Skipping {species}: missing reference FASTA {ref_fasta_path.name}")
        continue

    class_props_by_k, totals_by_k, n_samples, total_rows, used_rows = get_k_spectra(vcf_path, ref_fasta_path)
    species_results[species] = class_props_by_k
    species_site_counts[species] = totals_by_k

    print(f"Processed {species}")
    print(f"  Samples: {n_samples}")
    print(f"  Rows scanned: {total_rows}")
    print(f"  Rows used: {used_rows}")
    print(f"  Sites per k: {totals_by_k}")

species_names = sorted(species_results.keys(), key=lambda s: format_species_label(s))

if len(species_names) == 0:
    raise ValueError("No species were processed. Check VCF_DIR, REF_DIR, and filename matching.")

# Reorganise for plotting
class_to_matrix = {}
for cls in CLASSES:
    arr = np.zeros((len(species_names), len(KS)), dtype=float)
    for si, species in enumerate(species_names):
        for ki, k in enumerate(KS):
            arr[si, ki] = species_results[species][k][cls]
    class_to_matrix[cls] = arr

# Colour setup
cmap = plt.colormaps["turbo"]
n_species = len(species_names)

species_base_colors = {
    species: cmap(0.10 + 0.80 * (i / (n_species - 1 if n_species > 1 else 1)))
    for i, species in enumerate(species_names)
}

species_k_colors = {
    species: make_k_shades(species_base_colors[species])
    for species in species_names
}

# Global y-limit
global_max = max(class_to_matrix[cls].max() for cls in CLASSES)
ymax = global_max * 1.15 if global_max > 0 else 1.0

# Legend handles
species_handles = [
    Patch(facecolor=species_base_colors[s], edgecolor="black", linewidth=0.35)
    for s in species_names
]

species_labels = [format_species_label(s) for s in species_names]

example_base = "#4E79A7"
example_shades = make_k_shades(example_base)

k_handles = [
    Patch(facecolor=example_shades[1], edgecolor="black", linewidth=0.35),
    Patch(facecolor=example_shades[2], edgecolor="black", linewidth=0.35),
    Patch(facecolor=example_shades[3], edgecolor="black", linewidth=0.35),
]

# Plot
fig, axes = plt.subplots(3, 2, figsize=(18, 12), sharex=True, sharey=True)
axes = axes.flatten()

x = np.arange(len(species_names))
n_k = len(KS)
group_width = 0.78
bar_width = group_width / n_k

for ax, cls in zip(axes, PANEL_ORDER):
    arr = class_to_matrix[cls]
    for ki, k in enumerate(KS):
        offset = (ki - (n_k - 1) / 2) * bar_width
        bar_colors = [species_k_colors[species][k] for species in species_names]
        ax.bar(x + offset, arr[:, ki], width=bar_width, color=bar_colors, edgecolor="black", linewidth=0.35, zorder=3)
    ax.set_title(cls)
    ax.set_ylabel("Proportion of mutations")
    ax.set_ylim(0, ymax)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)

for ax in axes[:4]:
    ax.set_xticks(x)
    ax.set_xticklabels([])

for ax in axes[4:]:
    ax.set_xticks(x)
    apply_species_xticklabels(ax, species_names, rotation=60, fontsize=9)

fig.suptitle("Six-class mutation spectra across species and allele frequency", fontsize=18, y=0.985)

for ax in axes:
    ax.tick_params(labelleft=True)

fig.legend(
    handles=k_handles,
    labels=["Singletons (k=1)", "Doubletons (k=2)", "Tripletons (k=3)"],
    title="Allele frequency class (light → dark within each species colour)",
    loc="lower center",
    bbox_to_anchor=(0.5, 0.03),
    ncol=3,
    frameon=False,
    fontsize=10,
    title_fontsize=10
)

plt.tight_layout(rect=[0.03, 0.18, 0.97, 0.95])
plt.show()
