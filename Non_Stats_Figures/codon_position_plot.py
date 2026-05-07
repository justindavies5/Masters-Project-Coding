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

def build_codon_position_masks(ref_seqs, cds_features):
    # Mask values: 0 = non-CDS, 1/2/3 = codon position, -1 = ambiguous
    codon_mask = {}
    for seqid, seq in ref_seqs.items():
        codon_mask[seqid] = np.zeros(len(seq) + 1, dtype=np.int8)

    for seqid, start, end, strand, phase in cds_features:
        if seqid not in codon_mask:
            continue
        n = len(ref_seqs[seqid])
        start = max(1, start)
        end = min(n, end)
        if start > end:
            continue
        if strand == "+":
            for pos in range(start, end + 1):
                offset = pos - start
                codon_pos = ((offset + phase) % 3) + 1
                current = codon_mask[seqid][pos]
                if current == 0:
                    codon_mask[seqid][pos] = codon_pos
                elif current != codon_pos:
                    codon_mask[seqid][pos] = -1
        else:
            for pos in range(start, end + 1):
                offset = end - pos
                codon_pos = ((offset + phase) % 3) + 1
                current = codon_mask[seqid][pos]
                if current == 0:
                    codon_mask[seqid][pos] = codon_pos
                elif current != codon_pos:
                    codon_mask[seqid][pos] = -1

    pos1_bases = 0
    pos2_bases = 0
    pos3_bases = 0
    ambiguous_bases = 0
    total_bases = 0

    for seqid, seq in ref_seqs.items():
        mask = codon_mask[seqid]
        for pos in range(1, len(seq) + 1):
            if seq[pos - 1] not in BASES:
                continue
            total_bases += 1
            val = mask[pos]
            if val == 1:
                pos1_bases += 1
            elif val == 2:
                pos2_bases += 1
            elif val == 3:
                pos3_bases += 1
            elif val == -1:
                ambiguous_bases += 1

    return codon_mask, pos1_bases, pos2_bases, pos3_bases, ambiguous_bases, total_bases

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
def analyse_species_codon_positions(vcf_path: Path, fasta_path: Path, gff_path: Path):
    ref_seqs = read_fasta_dict(fasta_path)
    cds_features = parse_gff_cds_features(gff_path, feature_types=FEATURE_TYPES)
    codon_mask, pos1_bases, pos2_bases, pos3_bases, ambiguous_bases, total_bases = build_codon_position_masks(
        ref_seqs, cds_features
    )

    pos1 = 0
    pos2 = 0
    pos3 = 0
    total_rows = 0
    used_rows = 0
    skipped_all_ref = 0
    skipped_not_k1 = 0
    skipped_missing_seq = 0
    skipped_bad_pos = 0
    skipped_noncoding = 0
    skipped_ambiguous = 0

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
            if chrom not in ref_seqs or chrom not in codon_mask:
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
            codon_val = codon_mask[chrom][pos]
            if codon_val == 0:
                skipped_noncoding += 1
                continue
            elif codon_val == -1:
                skipped_ambiguous += 1
                continue
            elif codon_val == 1:
                pos1 += 1
            elif codon_val == 2:
                pos2 += 1
            elif codon_val == 3:
                pos3 += 1
            used_rows += 1

    raw_total = pos1 + pos2 + pos3
    raw_pos1_prop = pos1 / raw_total if raw_total > 0 else 0.0
    raw_pos2_prop = pos2 / raw_total if raw_total > 0 else 0.0
    raw_pos3_prop = pos3 / raw_total if raw_total > 0 else 0.0

    norm_pos1 = pos1 / pos1_bases if pos1_bases > 0 else 0.0
    norm_pos2 = pos2 / pos2_bases if pos2_bases > 0 else 0.0
    norm_pos3 = pos3 / pos3_bases if pos3_bases > 0 else 0.0

    norm_total = norm_pos1 + norm_pos2 + norm_pos3
    norm_pos1_prop = norm_pos1 / norm_total if norm_total > 0 else 0.0
    norm_pos2_prop = norm_pos2 / norm_total if norm_total > 0 else 0.0
    norm_pos3_prop = norm_pos3 / norm_total if norm_total > 0 else 0.0

    ratio_1_2 = norm_pos1 / norm_pos2 if norm_pos2 > 0 else float("inf")
    ratio_1_3 = norm_pos1 / norm_pos3 if norm_pos3 > 0 else float("inf")
    ratio_2_3 = norm_pos2 / norm_pos3 if norm_pos3 > 0 else float("inf")

    return {
        "total_rows": total_rows,
        "used_rows": used_rows,
        "skipped_all_ref": skipped_all_ref,
        "skipped_not_k1": skipped_not_k1,
        "skipped_missing_seq": skipped_missing_seq,
        "skipped_bad_pos": skipped_bad_pos,
        "skipped_noncoding": skipped_noncoding,
        "skipped_ambiguous": skipped_ambiguous,
        "pos1_bases": pos1_bases,
        "pos2_bases": pos2_bases,
        "pos3_bases": pos3_bases,
        "ambiguous_bases": ambiguous_bases,
        "total_bases": total_bases,
        "raw_pos1": pos1,
        "raw_pos2": pos2,
        "raw_pos3": pos3,
        "raw_pos1_prop": raw_pos1_prop,
        "raw_pos2_prop": raw_pos2_prop,
        "raw_pos3_prop": raw_pos3_prop,
        "norm_pos1": norm_pos1,
        "norm_pos2": norm_pos2,
        "norm_pos3": norm_pos3,
        "norm_pos1_prop": norm_pos1_prop,
        "norm_pos2_prop": norm_pos2_prop,
        "norm_pos3_prop": norm_pos3_prop,
        "ratio_1_2": ratio_1_2,
        "ratio_1_3": ratio_1_3,
        "ratio_2_3": ratio_2_3,
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

    res = analyse_species_codon_positions(vcf_path, fasta_path, gff_path)
    results[species] = res

    print(f"Processed {species}")
    print(f"  norm pos1 rate = {res['norm_pos1']:.6e}")
    print(f"  norm pos2 rate = {res['norm_pos2']:.6e}")
    print(f"  norm pos3 rate = {res['norm_pos3']:.6e}")

if len(results) == 0:
    raise ValueError("No species were processed. Check paths and filename matching.")

species_names = sorted(results.keys())
display_names = [format_species_label(s) for s in species_names]

# Table
print("\nCodon-position summary table:\n")

SPECIES_WIDTH = max(len(format_species_label(s)) for s in species_names) + 2

header = (
    f"{'Species':<{SPECIES_WIDTH}}"
    f"{'Raw_pos1':>10}"
    f"{'Raw_pos2':>10}"
    f"{'Raw_pos3':>10}"
    f"{'Pos1_bases':>12}"
    f"{'Pos2_bases':>12}"
    f"{'Pos3_bases':>12}"
    f"{'Norm_pos1':>14}"
    f"{'Norm_pos2':>14}"
    f"{'Norm_pos3':>14}"
    f"{'1:2':>10}"
    f"{'1:3':>10}"
    f"{'2:3':>10}"
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
        f"{r['raw_pos1']:10d}"
        f"{r['raw_pos2']:10d}"
        f"{r['raw_pos3']:10d}"
        f"{r['pos1_bases']:12d}"
        f"{r['pos2_bases']:12d}"
        f"{r['pos3_bases']:12d}"
        f"{r['norm_pos1']:14.6e}"
        f"{r['norm_pos2']:14.6e}"
        f"{r['norm_pos3']:14.6e}"
        f"{fmt_ratio(r['ratio_1_2']):>10}"
        f"{fmt_ratio(r['ratio_1_3']):>10}"
        f"{fmt_ratio(r['ratio_2_3']):>10}"
    )

# Colours
master_species_order = sorted(results.keys())
cmap = plt.colormaps[COLORMAP_NAME]
n_species = len(master_species_order)

species_colors = {
    species: cmap(0.35 + 0.5 * (i / (n_species - 1 if n_species > 1 else 1)))
    for i, species in enumerate(master_species_order)
}

# Order
species_order = sorted(results.keys(), key=lambda s: format_species_label(s))
display_names_ordered = [format_species_label(s) for s in species_order]
bar_colors = [species_colors[s] for s in species_order]

pos1_vals = [results[s]["norm_pos1_prop"] for s in species_order]
pos2_vals = [results[s]["norm_pos2_prop"] for s in species_order]
pos3_vals = [results[s]["norm_pos3_prop"] for s in species_order]

# Y limit
global_max = max(
    max(pos1_vals) if pos1_vals else 0,
    max(pos2_vals) if pos2_vals else 0,
    max(pos3_vals) if pos3_vals else 0
)
ymax = global_max * 1.15 if global_max > 0 else 1.0

# Plot
fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
x = np.arange(len(species_order))

axes[0].bar(x, pos1_vals, color=bar_colors, edgecolor="black", linewidth=0.6, zorder=3)
axes[0].set_title("Codon position 1")
axes[0].set_ylabel("Proportion of mutations")
axes[0].grid(axis="y", alpha=0.25, linewidth=0.6)
axes[0].set_axisbelow(True)
axes[0].set_ylim(0, ymax)

axes[1].bar(x, pos2_vals, color=bar_colors, edgecolor="black", linewidth=0.6, zorder=3)
axes[1].set_title("Codon position 2")
axes[1].set_ylabel("Proportion of mutations")
axes[1].grid(axis="y", alpha=0.25, linewidth=0.6)
axes[1].set_axisbelow(True)
axes[1].set_ylim(0, ymax)

axes[2].bar(x, pos3_vals, color=bar_colors, edgecolor="black", linewidth=0.6, zorder=3)
axes[2].set_title("Codon position 3")
axes[2].set_ylabel("Proportion of mutations")
axes[2].set_xticks(x)
apply_species_xticklabels(axes[2], display_names_ordered, rotation=60, fontsize=9)
axes[2].grid(axis="y", alpha=0.25, linewidth=0.6)
axes[2].set_axisbelow(True)
axes[2].set_ylim(0, ymax)

panel_labels = ["a)", "b)", "c)"]
for ax, label in zip(axes, panel_labels):
    ax.text(-0.08, 1.05, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=14, fontweight="bold")

fig.suptitle("Codon-position mutation proportions across species", fontsize=16, y=0.99)
plt.tight_layout(rect=[0.05, 0, 1, 0.97])
plt.show()
