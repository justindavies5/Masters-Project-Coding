#Done relatively close to the end so both graphs were grouped together for time efficiency 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np

# Colours

COLOUR_SHADES = {
    "C>A": ["#aed6f1", "#2e86c1", "#1a3a5c"],
    "C>G": ["#fad7a0", "#e67e22", "#784212"],
    "C>T": ["#f1948a", "#cb4335", "#641e16"],
    "T>A": ["#d7bde2", "#7d3c98", "#4a235a"],
    "T>C": ["#a9dfbf", "#1e8449", "#0b5327"],
    "T>G": ["#fdebd0", "#d4ac0d", "#7d6608"],
}

CONTEXT_COLOUR = {mc: COLOUR_SHADES[mc][1] for mc in COLOUR_SHADES}

# 1. Allele frequency

df = pd.read_csv("allele_freq_combined_IRR_wald.csv")
df["pct_change"] = (df["estimate"] - 1) * 100

mut_order   = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]
comparisons = ["k1_vs_k2", "k1_vs_k3", "k2_vs_k3"]
comp_labels = {
    "k1_vs_k2": "k=1 vs k=2",
    "k1_vs_k3": "k=1 vs k=3",
    "k2_vs_k3": "k=2 vs k=3",
}
SHADE_IDX = {"k1_vs_k2": 0, "k1_vs_k3": 1, "k2_vs_k3": 2}

REMAP = {
    "k2_vs_k1": "k1_vs_k2",
    "k3_vs_k1": "k1_vs_k3",
    "k3_vs_k2": "k2_vs_k3",
}
df["comparison"] = df["comparison"].replace(REMAP)
df = df[df["mut_class"].isin(mut_order)].copy()
df["mut_class"] = pd.Categorical(df["mut_class"],
                                  categories=mut_order, ordered=True)

bar_width = 0.25
group_gap = 0.12
n_comp    = len(comparisons)
n_mut     = len(mut_order)

max_abs = 0
for comp in comparisons:
    sub = df[df["comparison"] == comp]
    if not sub.empty:
        max_abs = max(max_abs, sub["pct_change"].abs().max())
y_lim = max_abs * 1.25

fig, ax = plt.subplots(figsize=(14, 6))

for ci, comp in enumerate(comparisons):
    sub = (df[df["comparison"] == comp]
           .sort_values("mut_class")
           .set_index("mut_class"))

    for mi, mc in enumerate(mut_order):
        xi    = mi * (n_comp * bar_width + group_gap) + ci * bar_width
        shade = COLOUR_SHADES[mc][SHADE_IDX[comp]]

        if mc not in sub.index:
            continue

        row = sub.loc[mc]
        val = row["pct_change"]
        sig = row["signif"]

        ax.bar(xi, val, width=bar_width,
               color=shade, edgecolor="black", linewidth=0.5)

        if sig != "ns":
            offset = y_lim * 0.02
            y_pos  = val - offset if val < 0 else val + offset
            va     = "top" if val < 0 else "bottom"
            ax.text(xi, y_pos, sig,
                    ha="center", va=va,
                    fontsize=8, color="black")

group_centres = [
    mi * (n_comp * bar_width + group_gap) + (n_comp * bar_width) / 2
    for mi in range(n_mut)
]
ax.set_xticks(group_centres)
ax.set_xticklabels(mut_order, fontsize=10, rotation=45, ha="right")
ax.set_ylim(-y_lim, y_lim)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xlabel("Mutation class", fontsize=10)
ax.set_ylabel("Percent change in mutation frequency (%)", fontsize=10)
ax.set_title("Shift in mutation spectrum across allele frequency classes",
             fontsize=11)

ax.text(0.98, 0.97, "Higher in first-listed k class",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=9, color="black")
ax.text(0.98, 0.03, "Lower in first-listed k class",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9, color="black")

legend_patches = []
ref_mc = "C>A"
for comp in comparisons:
    swatch = COLOUR_SHADES[ref_mc][SHADE_IDX[comp]]
    legend_patches.append(
        mpatches.Patch(facecolor=swatch, edgecolor="black",
                       linewidth=0.5, label=comp_labels[comp])
    )
ax.legend(handles=legend_patches, fontsize=9,
          title="Comparison", title_fontsize=9,
          loc="upper left", framealpha=0.9)

plt.tight_layout()
plt.show()


# 2. Local Sequence Context

WSPACE         = 0.11
HSPACE         = 0.20
TICK_FONTSIZE  = 10
LABEL_FONTSIZE = 10

df_ctx = pd.read_csv("context_combined_IRR_wald.csv")
df_ctx["pct_change"] = (df_ctx["estimate"] - 1) * 100

LEFT_COL  = ["C>A", "C>G", "C>T"]
RIGHT_COL = ["T>A", "T>G", "T>C"]

contexts = [
    "A_C", "A_G", "A_T",
    "C_A", "C_C", "C_G", "C_T",
    "G_A", "G_C", "G_G", "G_T",
    "T_A", "T_C", "T_G", "T_T"
]

global_max = 0
for mc in LEFT_COL + RIGHT_COL:
    sub = df_ctx[
        (df_ctx["mut_class"] == mc) &
        (df_ctx["comparison"].str.endswith("_vs_A_A"))
    ]
    if not sub.empty:
        global_max = max(global_max, sub["pct_change"].abs().max())
y_lim_ctx = global_max * 1.15

fig = plt.figure(figsize=(18, 12))
gs  = gridspec.GridSpec(
    3, 2,
    hspace=HSPACE,
    wspace=WSPACE,
    left=0.06,
    right=0.98,
    top=0.93,
    bottom=0.12
)

for row_idx in range(3):
    mc_left  = LEFT_COL[row_idx]
    mc_right = RIGHT_COL[row_idx]

    ax_left  = fig.add_subplot(gs[row_idx, 0])
    ax_right = fig.add_subplot(gs[row_idx, 1], sharey=ax_left)

    for ax, mc, is_right in [(ax_left, mc_left, False),
                              (ax_right, mc_right, True)]:

        sub = df_ctx[
            (df_ctx["mut_class"] == mc) &
            (df_ctx["comparison"].str.endswith("_vs_A_A"))
        ].copy()

        sub["context"] = sub["comparison"].str.replace(
            "_vs_A_A", "", regex=False
        )
        sub["context"] = pd.Categorical(
            sub["context"], categories=contexts, ordered=True
        )
        sub = sub.sort_values("context")

        x   = np.arange(len(contexts))
        col = CONTEXT_COLOUR[mc]

        ax.bar(x, sub["pct_change"].values, width=0.7,
               color=col, edgecolor="black", linewidth=0.4)

        for xi, (_, row) in enumerate(sub.iterrows()):
            sig = row["signif"]
            if sig != "ns":
                val    = row["pct_change"]
                offset = y_lim_ctx * 0.02
                y_pos  = val - offset if val < 0 else val + offset
                va     = "top" if val < 0 else "bottom"
                ax.text(xi, y_pos, sig,
                        ha="center", va=va,
                        fontsize=7, color="black")

        ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
        ax.set_ylim(-y_lim_ctx, y_lim_ctx)

        ax.set_title(mc, fontsize=11, fontweight="normal",
                     color="black", pad=4)

        ax.set_xticks(x)
        ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)

        if row_idx == 2:
            ax.set_xticklabels(contexts, rotation=60, ha="right",
                               fontsize=TICK_FONTSIZE)
            ax.set_xlabel("Local sequence context",
                          fontsize=LABEL_FONTSIZE)
        else:
            ax.set_xticklabels([])
            ax.set_xlabel("")

        ax.tick_params(labelleft=True)
        ax.yaxis.set_tick_params(labelsize=TICK_FONTSIZE)
        ax.set_ylabel("Percent change relative to A_A context (%)",
                      fontsize=LABEL_FONTSIZE)

fig.suptitle(
    "Mutation frequency by local sequence context relative to A_A baseline",
    fontsize=13, fontweight="normal", color="black"
)

plt.show()