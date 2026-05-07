import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

# 1. Load CSV
csv_path = "/shared/team/2025-masters-project/people/justin/combined_IRR_tables_with_stats/synonymous_nonsynonymous_combined_IRR_wald.csv"
df = pd.read_csv(csv_path)

# 2. Check columns
required_cols = ["mut_class", "estimate", "p.value"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# 3. Add significance stars
def get_signif(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"

df["signif"] = df["p.value"].apply(get_signif)

# 4. Mutation order
mut_order = [
    "A>C", "A>G", "A>T",
    "C>A", "C>G", "C>T",
    "G>A", "G>C", "G>T",
    "T>A", "T>C", "T>G"
]

df["mut_class"] = pd.Categorical(df["mut_class"], categories=mut_order, ordered=True)
df = df.sort_values("mut_class").reset_index(drop=True)

# 5. Convert to % change
df["percent_change"] = (df["estimate"] - 1.0) * 100.0

# 6. Ref-base pastel gradient colours
def lighten_color(color, amount):
    c = np.array(to_rgb(color))
    white = np.array([1, 1, 1])
    return tuple(c + (white - c) * amount)

base_colors = {
    "A": "#f94449", 
    "C": "#e6cc00", 
    "G": "#8bca84",  
    "T": "#0096c7", 
}

gradient_levels = [0.00, 0.18, 0.36]

mutation_color_map = {}
for ref in ["A", "C", "G", "T"]:
    muts_for_ref = [m for m in mut_order if m.startswith(ref)]
    for i, mut in enumerate(muts_for_ref):
        mutation_color_map[mut] = lighten_color(base_colors[ref], gradient_levels[i])

bar_colors = [mutation_color_map[m] for m in df["mut_class"]]

# 7. Plot
fig, ax = plt.subplots(figsize=(12, 7))

x = np.arange(len(df))
y = df["percent_change"].values

bars = ax.bar(
    x,
    y,
    width=0.75,
    color=bar_colors,
    edgecolor="black",
    linewidth=0.8
)

ax.axhline(0, color="black", linewidth=1.2, linestyle=":")

# 8. Significance labels
y_abs = max(abs(y.min()), abs(y.max()))
offset = max(1.5, y_abs * 0.04)

for bar, sig, val in zip(bars, df["signif"], y):
    x_pos = bar.get_x() + bar.get_width() / 2

    if val >= 0:
        ax.text(
            x_pos, val + offset, sig,
            ha="center", va="bottom", fontsize=11
        )
    else:
        ax.text(
            x_pos, val - offset, sig,
            ha="center", va="top", fontsize=11
        )

# 9. Regular labels
ax.set_xticks(x)
ax.set_xticklabels(df["mut_class"], rotation=45, ha="right")

ax.set_ylabel("Percent change (non-synonymous vs synonymous) (%)")
ax.set_xlabel("Mutation class")
ax.set_title("Non-synonymous vs synonymous mutation frequency")

# 10. Interpretation labels
ax.text(
    len(df)-0.2, y_abs*0.98,
    "Higher in non-synonymous",
    ha="right", va="top", fontsize=10
)

ax.text(
    len(df)-0.2, -y_abs*0.98,
    "Higher in synonymous",
    ha="right", va="bottom", fontsize=10
)

ylim = y_abs + 3 * offset
ax.set_ylim(-ylim, ylim)

plt.tight_layout()
plt.show()