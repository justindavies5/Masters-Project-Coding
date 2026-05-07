import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path

# Settings
VCF_DIR = Path("/shared/team/2025-masters-project/people/justin/all_genomes_singletonsandwindowmasked")
REF_DIR = Path("/shared/team/2025-masters-project/people/justin/all_genomes_referenceandFFA_files")

ALLOW_FILTER = {"PASS", "."}
CLASSES = ["C>A", "C>G", "C>T", "T>A", "T>G", "T>C"]

COLORS = {
    "C>T": "#E15759",
    "C>A": "#F28E2B",
    "C>G": "#59A14F",
    "T>C": "#4E79A7",
    "T>A": "#76B7B2",
    "T>G": "#EDC948",
}

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

def safe_norm(values_dict):
    total = sum(values_dict.values())
    if total == 0:
        return {k: 0.0 for k in values_dict}
    return {k: v / total for k, v in values_dict.items()}

def species_name_from_vcf(vcf_path: Path) -> str:
    suffix = "_biallelic_k1_singletons_windowmasked.vcf"
    name = vcf_path.name
    if name.endswith(suffix):
        return name[:-len(suffix)]
    return vcf_path.stem

# Process one species
def get_normalised_6class_spectrum(vcf_path: Path, ref_fasta_path: Path):
    directional_counts = Counter()
    total_k1 = 0

    with open(vcf_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
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
                    if k > 1:
                        break
            if k != 1:
                continue
            ref = col[3].upper()
            alt = col[4].upper()
            directional_counts[(ref, alt)] += 1
            total_k1 += 1

    ref_counts = read_fasta_base_counts(ref_fasta_path)

    directional_rates = {}
    for ref, alt in directional_counts:
        opp = ref_counts[ref]
        directional_rates[(ref, alt)] = directional_counts[(ref, alt)] / opp if opp > 0 else 0.0

    collapsed_rates = {cls: 0.0 for cls in CLASSES}
    for (ref, alt), rate in directional_rates.items():
        cls = to_6class(ref, alt)
        collapsed_rates[cls] += rate

    plot_vals = safe_norm(collapsed_rates)
    return plot_vals, total_k1

# Match files and run
vcf_files = sorted(VCF_DIR.glob("*_biallelic_k1_singletons_windowmasked.vcf"))

species_results = {}
species_k1_totals = {}

for vcf_path in vcf_files:
    species = species_name_from_vcf(vcf_path)
    ref_fasta_path = REF_DIR / f"{species}.fna"

    if not ref_fasta_path.exists():
        print(f"Skipping {species}: no matching reference FASTA found at {ref_fasta_path}")
        continue

    plot_vals, total_k1 = get_normalised_6class_spectrum(vcf_path, ref_fasta_path)
    species_results[species] = plot_vals
    species_k1_totals[species] = total_k1

    print(f"Processed {species}")
    print(f"  VCF : {vcf_path.name}")
    print(f"  REF : {ref_fasta_path.name}")
    print(f"  k=1 : {total_k1}")
    print(f"  spectrum: {plot_vals}")

# Prepare data for plotting
species_names = sorted(species_results.keys(), key=lambda s: format_species_label(s))

if len(species_names) == 0:
    raise ValueError("No species were processed. Check your VCF_DIR, REF_DIR, and filenames.")

class_to_values = {
    cls: [species_results[species][cls] for species in species_names]
    for cls in CLASSES
}

# Colour setup
cmap = plt.colormaps["plasma"]
n_species = len(species_names)

species_colors = {
    species: cmap(0.35 + 0.5 * (i / (n_species - 1 if n_species > 1 else 1)))
    for i, species in enumerate(species_names)
}

# Global y-limit
global_max = max(max(class_to_values[cls]) for cls in CLASSES)
ymax = global_max * 1.15 if global_max > 0 else 1.0

# Panel order
panel_order = ["C>A", "T>A", "C>G", "T>G", "C>T", "T>C"]

# Plot
fig, axes = plt.subplots(3, 2, figsize=(18, 12), sharex=True)
axes = axes.flatten()

x = np.arange(len(species_names))
bar_colors = [species_colors[s] for s in species_names]

for ax, cls in zip(axes, panel_order):
    y = class_to_values[cls]
    ax.bar(x, y, color=bar_colors, edgecolor="black", linewidth=0.6, zorder=3)
    ax.set_title(cls)
    ax.set_ylabel("Proportion of mutations")
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_ylim(0, ymax)
    ax.set_xticks(x)

for ax in axes[:4]:
    ax.set_xticklabels([])

apply_species_xticklabels(axes[4], species_names, rotation=60, fontsize=9)
apply_species_xticklabels(axes[5], species_names, rotation=60, fontsize=9)

fig.suptitle("Composition-normalised six-class mutation spectra across species", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()
