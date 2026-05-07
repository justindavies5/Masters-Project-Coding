# Must be run after the six-class mutation spectra plot

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Data from notebook output
# (C>T:T>G ratio, dN/dS, genic_enrichment_ratio, group)
species = {
    "C. jejuni":          (44.08, 0.5666, 6.5046, "deamination"),
    "N. meningitidis":    (16.68, 0.4945, 1.8905, "deamination"),
    "K. pneumoniae":      ( 9.30, 0.4118, 1.2803, "intermediate"),
    "S. epidermidis":     (16.57, 0.7432, 4.1326, "intermediate"),
    "P. aeruginosa":      ( 2.77, 0.5441, 1.0165, "oxidative"),
    "M. tuberculosis":    ( 3.19, 0.6888, 0.9432, "oxidative"),
    "L. monocytogenes":   (11.50, 0.5131, 1.5379, "intermediate"),
    "S. pneumoniae":      (11.74, 0.5174, 2.2297, "deamination"),
    "S. aureus":          (17.35, 0.7138, 3.7788, "intermediate"),
    "B. pertussis":       ( 3.20, 0.8384, 0.6151, "drift"),
    "S. agalactiae":      (17.34, 0.7792, 1.3712, "intermediate"),
    "E. coli":            ( 9.45, 0.4681, 1.1686, "intermediate"),
    "H. influenzae":      (12.40, 0.5482, 2.7662, "intermediate"),
    "S. typhimurium":     ( 4.34, 0.7357, 0.7012, "oxidative"),
}

group_colours = {
    "deamination":  "#2166AC",
    "oxidative":    "#D6604D",
    "drift":        "#F4A582",
    "intermediate": "#969696",
}

group_labels = {
    "deamination":  "Deamination-driven",
    "oxidative":    "Oxidative-driven",
    "drift":        "Oxidative or drift-driven",
    "intermediate": "Ambiguous",
}

group_order = ["deamination", "oxidative", "drift", "intermediate"]

# Genic enrichment → point size (sqrt scaled)
genic_vals = np.array([v[2] for v in species.values()])
size_min, size_max = 80, 420
ge_sqrt = np.sqrt(genic_vals)
ge_norm = (ge_sqrt - ge_sqrt.min()) / (ge_sqrt.max() - ge_sqrt.min())
sizes = size_min + ge_norm * (size_max - size_min)

# Mean reference lines
x_vals = np.array([v[0] for v in species.values()])
y_vals = np.array([v[1] for v in species.values()])
mean_x = np.mean(x_vals)
mean_y = np.mean(y_vals)

# Label offsets (dx, dy, ha)
label_offsets = {
    "C. jejuni":          ( 0.5,   0.022, "left"),
    "N. meningitidis":    ( 0.5,   0.019, "left"),
    "K. pneumoniae":      ( 0.5,   0.019, "left"),
    "S. epidermidis":     (-0.5,  -0.022, "right"),
    "P. aeruginosa":      ( 0.5,  -0.022, "left"),
    "M. tuberculosis":    ( 0.5,  -0.022, "left"),
    "L. monocytogenes":   (-0.5,   0.022, "right"),
    "S. pneumoniae":      ( 0.5,  -0.022, "left"),
    "S. aureus":          ( 0.5,   0.022, "left"),
    "B. pertussis":       ( 0.5,   0.022, "left"),
    "S. agalactiae":      ( 0.5,   0.022, "left"),
    "E. coli":            ( 0.5,  -0.022, "left"),
    "H. influenzae":      ( 0.5,   0.022, "left"),
    "S. typhimurium":     ( 0.5,  -0.022, "left"),
}

def format_name(name):
    parts = name.split()
    return f"$\\it{{{parts[0][0]}}}$.$\\it{{{' '.join(parts[1:])}}}$"

# Figure
fig, ax = plt.subplots(figsize=(13, 8.5))
fig.patch.set_facecolor("white")
ax.set_facecolor("#FAFAFA")

ax.grid(True, linestyle="--", linewidth=0.5, color="#CCCCCC", alpha=0.7, zorder=0)

ax.axvspan(0,       mean_x, ymin=0, ymax=1, alpha=0.03, color="#D6604D", zorder=0)
ax.axvspan(mean_x, 50,     ymin=0, ymax=1, alpha=0.03, color="#2166AC", zorder=0)

ax.axvline(x=mean_x, color="#888888", linestyle=":", linewidth=1.1, zorder=1, alpha=0.8)
ax.axhline(y=mean_y, color="#888888", linestyle=":", linewidth=1.1, zorder=1, alpha=0.8)

ax.text(mean_x + 0.4, 0.393,
        f"Mean C>T:T>G = {mean_x:.2f}",
        fontsize=7.5, color="#666666", va="bottom", style="italic")
ax.text(0.3, mean_y + 0.007,
        f"Mean dN/dS = {mean_y:.2f}",
        fontsize=7.5, color="#666666", ha="left", style="italic")

# Plot points
for (name, (ratio, dnds, genic, group)), size in zip(species.items(), sizes):
    ax.scatter(ratio, dnds,
               s=size, color=group_colours[group],
               edgecolors="white", linewidths=0.9,
               alpha=0.90, zorder=3)

# Labels
for name, (ratio, dnds, genic, group) in species.items():
    dx, dy, ha = label_offsets[name]
    ax.annotate(
        format_name(name),
        xy=(ratio, dnds),
        xytext=(ratio + dx, dnds + dy),
        fontsize=8.2, ha=ha, va="center",
        color="#222222",
        arrowprops=dict(arrowstyle="-", color="#BBBBBB", lw=0.55),
        zorder=4
    )

# Legends
group_patches = [
    mpatches.Patch(facecolor=group_colours[g], edgecolor="white", label=group_labels[g])
    for g in group_order
]

ref_ge = [genic_vals.min(), np.median(genic_vals), genic_vals.max()]
ref_labels = [
    f"Low  (ratio ≈ {genic_vals.min():.1f})",
    f"Medium  (ratio ≈ {np.median(genic_vals):.1f})",
    f"High  (ratio ≈ {genic_vals.max():.1f})",
]
ref_sizes = [
    size_min + ((np.sqrt(v) - ge_sqrt.min()) /
                (ge_sqrt.max() - ge_sqrt.min())) * (size_max - size_min)
    for v in ref_ge
]
size_handles = [
    plt.scatter([], [], s=s, color="#777777", edgecolors="white", linewidths=0.9, alpha=0.90, label=l)
    for s, l in zip(ref_sizes, ref_labels)
]

leg1 = ax.legend(
    handles=group_patches,
    title="Mutational profile",
    title_fontsize=9, fontsize=8.5,
    loc="lower left",
    bbox_to_anchor=(0.0, -0.22),
    ncol=4,
    framealpha=0.95, edgecolor="#CCCCCC",
    borderpad=0.8,
)
ax.add_artist(leg1)

ax.legend(
    handles=size_handles,
    title="Genic enrichment ratio",
    title_fontsize=9, fontsize=8.5,
    loc="lower right",
    bbox_to_anchor=(1.0, -0.22),
    ncol=3,
    framealpha=0.95, edgecolor="#CCCCCC",
    borderpad=0.8,
)

# Axes
ax.set_xlabel("C>T : T>G ratio", fontsize=11, labelpad=8)
ax.set_ylabel("dN/dS ratio", fontsize=12, labelpad=8)
ax.set_title("Mutational profile space across 14 pathogenic bacterial species",
             fontsize=13, fontweight="bold", pad=14)

ax.set_xlim(0, 50)
ax.set_ylim(0.38, 0.88)
ax.tick_params(axis="both", labelsize=9)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/species_mutational_profile_CT_TG_final.png",
            dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig("/mnt/user-data/outputs/species_mutational_profile_CT_TG_final.pdf",
            bbox_inches="tight", facecolor="white")
print(f"Mean C>T:T>G = {mean_x:.3f}")
print(f"Mean dN/dS   = {mean_y:.3f}")
print("Saved.")
