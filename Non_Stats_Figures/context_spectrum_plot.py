import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import Counter
from pathlib import Path

# Settings
VCF_DIR = Path("/shared/team/2025-masters-project/people/justin/all_genomes_singletonsandwindowmasked")
REF_DIR = Path("/shared/team/2025-masters-project/people/justin/all_genomes_referenceandFFA_files")

VCF_SUFFIX = "_biallelic_k1_singletons_windowmasked.vcf"
REF_SUFFIX = ".fna"

ALLOW_FILTER = {"PASS", "."}
ENFORCE_K1 = True
HEADROOM = 1.05

BASES = {"A", "C", "G", "T"}
COMP = {"A": "T", "T": "A", "C": "G", "G": "C"}

SIX_CLASSES = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]
FLANK_BASES = ["A", "C", "G", "T"]
b2i = {b: i for i, b in enumerate(FLANK_BASES)}

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

def count_trinuc_opportunities_pyrimidine(seqs: dict) -> Counter:
    opp = Counter()
    for seq in seqs.values():
        n = len(seq)
        for i in range(1, n - 1):
            L = seq[i - 1]
            ref = seq[i]
            R = seq[i + 1]
            if L in BASES and ref in {"C", "T"} and R in BASES:
                opp[(L, ref, R)] += 1
    return opp

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

def reverse_complement_trinuc(L, ref, R):
    return (COMP[R], COMP[ref], COMP[L])

def to_pyrimidine_view(L, ref, alt, R):
    ref = ref.upper()
    alt = alt.upper()
    L = L.upper()
    R = R.upper()
    if ref in {"A", "G"}:
        L, ref, R = reverse_complement_trinuc(L, ref, R)
        alt = COMP[alt]
    return L, ref, alt, R

def species_name_from_vcf(vcf_path: Path) -> str:
    if vcf_path.name.endswith(VCF_SUFFIX):
        return vcf_path.name[:-len(VCF_SUFFIX)]
    return vcf_path.stem

# Process one species
def get_context_normalised_spectrum(vcf_path: Path, ref_fasta_path: Path):
    ref_seqs = read_fasta_dict(ref_fasta_path)
    opp_trinuc = count_trinuc_opportunities_pyrimidine(ref_seqs)

    counts_6x16 = {cls: Counter() for cls in SIX_CLASSES}

    with open(vcf_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            col = line.rstrip("\n").split("\t")
            if not is_biallelic_snp_pass(col):
                continue
            chrom = col[0]
            if chrom not in ref_seqs:
                continue
            seq = ref_seqs[chrom]
            pos1 = int(col[1])
            idx0 = pos1 - 1
            if idx0 - 1 < 0 or idx0 + 1 >= len(seq):
                continue
            L = seq[idx0 - 1]
            R = seq[idx0 + 1]
            ref = col[3].upper()
            alt = col[4].upper()
            if L not in BASES or R not in BASES:
                continue
            gt_idx = gt_index_from_format(col[8])
            if gt_idx is None:
                continue
            if ENFORCE_K1:
                carriers = 0
                for sf in col[9:]:
                    if is_alt_carrier(sf, gt_idx):
                        carriers += 1
                        if carriers > 1:
                            break
                if carriers != 1:
                    continue
            L2, ref2, alt2, R2 = to_pyrimidine_view(L, ref, alt, R)
            cls = f"{ref2}>{alt2}"
            if cls in counts_6x16:
                counts_6x16[cls][(L2, R2)] += 1

    raw_mats = {cls: np.zeros((4, 4), dtype=float) for cls in SIX_CLASSES}
    norm_mats = {cls: np.zeros((4, 4), dtype=float) for cls in SIX_CLASSES}
    plot_mats = {cls: np.zeros((4, 4), dtype=float) for cls in SIX_CLASSES}

    for cls in SIX_CLASSES:
        ref_base = cls[0]
        for L in FLANK_BASES:
            for R in FLANK_BASES:
                count = counts_6x16[cls][(L, R)]
                raw_mats[cls][b2i[L], b2i[R]] = count
                opp = opp_trinuc.get((L, ref_base, R), 0)
                norm_mats[cls][b2i[L], b2i[R]] = count / opp if opp > 0 else 0.0
        s = norm_mats[cls].sum()
        plot_mats[cls] = norm_mats[cls] / s if s > 0 else norm_mats[cls]

    return plot_mats

# Run all species
vcf_files = sorted(VCF_DIR.glob(f"*{VCF_SUFFIX}"))
species_plot_mats = {}

for vcf_path in vcf_files:
    species = species_name_from_vcf(vcf_path)
    ref_fasta_path = REF_DIR / f"{species}{REF_SUFFIX}"
    if not ref_fasta_path.exists():
        print(f"Skipping {species}: missing reference FASTA {ref_fasta_path.name}")
        continue
    plot_mats = get_context_normalised_spectrum(vcf_path, ref_fasta_path)
    species_plot_mats[species] = plot_mats
    print(f"Processed {species}")

species_names = sorted(species_plot_mats.keys())
display_names = [s.replace("_", " ") for s in species_names]

if len(species_names) == 0:
    raise ValueError("No species were processed. Check VCF_DIR, REF_DIR, and filename matching.")

# Reorganise data
contexts_16 = [(L, R) for L in FLANK_BASES for R in FLANK_BASES]

context_data = {}
for L, R in contexts_16:
    arr = np.zeros((len(species_names), len(SIX_CLASSES)), dtype=float)
    for si, species in enumerate(species_names):
        for ci, cls in enumerate(SIX_CLASSES):
            arr[si, ci] = species_plot_mats[species][cls][b2i[L], b2i[R]]
    context_data[(L, R)] = arr

global_max = max(context_data[(L, R)].max() for (L, R) in contexts_16)
ymax = global_max * 1.15 if global_max > 0 else 1.0

# Plot
def lighten_color(color, amount=0.5):
    c = np.array(mcolors.to_rgb(color))
    white = np.array([1.0, 1.0, 1.0])
    return tuple((1 - amount) * c + amount * white)

def make_species_shades(base_color, n_classes=6):
    amounts = np.linspace(0.55, 0.00, n_classes)
    return [lighten_color(base_color, amt) for amt in amounts]

cmap = plt.colormaps["turbo"]
n_species = len(species_names)

species_base_colors = {
    species: cmap(i / (n_species - 1 if n_species > 1 else 1))
    for i, species in enumerate(species_names)
}

species_class_colors = {
    species: make_species_shades(species_base_colors[species], n_classes=len(SIX_CLASSES))
    for species in species_names
}

fig, axes = plt.subplots(4, 4, figsize=(24, 18), sharex=True, sharey=True)
axes = axes.flatten()

x = np.arange(len(species_names))
n_classes = len(SIX_CLASSES)
group_width = 0.84
bar_width = group_width / n_classes

for ax, (L, R) in zip(axes, contexts_16):
    arr = context_data[(L, R)]
    for ci, cls in enumerate(SIX_CLASSES):
        offset = (ci - (n_classes - 1) / 2) * bar_width
        bar_colors = [species_class_colors[species][ci] for species in species_names]
        ax.bar(x + offset, arr[:, ci], width=bar_width, color=bar_colors, edgecolor="black", linewidth=0.35, zorder=3)
    ax.set_title(f"{L} _ {R}")
    ax.set_ylim(0, ymax)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)

for ax in axes[-4:]:
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=90)

for ax in axes[:-4]:
    ax.set_xticks(x)
    ax.set_xticklabels([])

for i in [0, 4, 8, 12]:
    axes[i].set_ylabel("Proportion within class")

# Legends
neutral_base = "#4C78A8"
neutral_shades = make_species_shades(neutral_base, n_classes=len(SIX_CLASSES))
mutation_handles = [
    plt.Rectangle((0, 0), 1, 1, color=neutral_shades[i], ec="black", lw=0.35, label=SIX_CLASSES[i])
    for i in range(len(SIX_CLASSES))
]

species_handles = [
    plt.Rectangle((0, 0), 1, 1, color=species_base_colors[s], ec="black", lw=0.35, label=display_names[i])
    for i, s in enumerate(species_names)
]

fig.suptitle("Context-normalised local sequence spectra across species", fontsize=18, y=0.995)

fig.legend(
    handles=mutation_handles,
    labels=SIX_CLASSES,
    ncol=6,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.965),
    frameon=False,
    title="Mutation class (light → dark shade within each species colour)"
)

fig.legend(
    handles=species_handles,
    labels=display_names,
    ncol=5,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.01),
    frameon=False,
    title="Species"
)

plt.tight_layout(rect=[0.02, 0.06, 1, 0.90])
plt.show()
