import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load CSV
csv_path = "/shared/team/2025-masters-project/people/justin/combined_IRR_tables_with_stats/codon_position_combined_IRR_wald2.csv"
df = pd.read_csv(csv_path)

# Check required columns
required_cols = ["mut_class", "comparison", "estimate", "signif"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Mutation class and comparison order
mut_order = [
    "A>C", "A>G", "A>T",
    "C>A", "C>G", "C>T",
    "G>A", "G>C", "G>T",
    "T>A", "T>C", "T>G"
]

comp_order = ["1_vs_2", "1_vs_3", "2_vs_3"]

df["mut_class"]  = pd.Categorical(df["mut_class"],  categories=mut_order,  ordered=True)
df["comparison"] = pd.Categorical(df["comparison"], categories=comp_order, ordered=True)
df = df.sort_values(["mut_class", "comparison"]).reset_index(drop=True)

# Convert to percent change
df["percent_change"] = (df["estimate"] - 1.0) * 100.0

# Pivot for grouped plotting
plot_df = df.pivot(index="mut_class", columns="comparison", values="percent_change")
sig_df  = df.pivot(index="mut_class", columns="comparison", values="signif")

# Plot settings
x         = np.arange(len(mut_order))
bar_width  = 0.22

comp_colors = {
    "1_vs_2": "#FBB4AE",
    "1_vs_3": "#B3CDE3",
    "2_vs_3": "#CCEBC5"
}

fig, ax = plt.subplots(figsize=(14, 8))

# Draw grouped bars
offsets = {
    "1_vs_2": -bar_width,
    "1_vs_3": 0,
    "2_vs_3": bar_width
}

bars_by_comp = {}

for comp in comp_order:
    y = plot_df[comp].values
    bars = ax.bar(
        x + offsets[comp],
        y,
        width=bar_width,
        label=comp.replace("_vs_", " vs "),
        color=comp_colors[comp],
        edgecolor="black",
        linewidth=0.8
    )
    bars_by_comp[comp] = bars

ax.axhline(0, color="black", linewidth=1.2, linestyle=":")

# Significance labels
all_y  = df["percent_change"].dropna().values
y_abs  = max(abs(all_y.min()), abs(all_y.max()))
offset = max(1.5, y_abs * 0.04)

for comp in comp_order:
    bars    = bars_by_comp[comp]
    yvals   = plot_df[comp].values
    sigvals = sig_df[comp].values

    for bar, val, sig in zip(bars, yvals, sigvals):
        if pd.isna(val) or pd.isna(sig):
            continue
        x_pos = bar.get_x() + bar.get_width() / 2
        if val >= 0:
            ax.text(x_pos, val + offset, str(sig), ha="center", va="bottom", fontsize=10)
        else:
            ax.text(x_pos, val - offset, str(sig), ha="center", va="top",    fontsize=10)

# Axes
ax.set_xticks(x)
ax.set_xticklabels(mut_order, rotation=45, ha="right")
ax.set_ylabel("Percent change in mutation frequency (%)")
ax.set_xlabel("Mutation class")
ax.set_title("Codon position comparison by mutation class")

ax.text(len(mut_order) - 0.15,  y_abs * 0.95, "Higher in first-listed codon position", ha="right", va="top",    fontsize=10)
ax.text(len(mut_order) - 0.15, -y_abs * 0.95, "Lower in first-listed codon position",  ha="right", va="bottom", fontsize=10)

ylim = y_abs + 3 * offset
ax.set_ylim(-ylim, ylim)
ax.legend(title="Comparison", frameon=False)

plt.tight_layout()
plt.show()
