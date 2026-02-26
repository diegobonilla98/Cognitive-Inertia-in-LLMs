import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

results_path = "results_scores.csv"
output_path = "eval_plot.png"

df = pd.read_csv(results_path)
df_long = df.melt(
    id_vars=["subject", "level"],
    value_vars=["smart_score", "stupid_score"],
    var_name="model",
    value_name="score",
)
df_long["model"] = df_long["model"].str.replace("_score", "").str.replace("_", " ").str.title()

n = len(df)
smart_mean = df["smart_score"].mean()
stupid_mean = df["stupid_score"].mean()
smart_std = df["smart_score"].std()
stupid_std = df["stupid_score"].std()
delta_mean = smart_mean - stupid_mean
summary_lines = [
    f"N = {n}",
    f"Smart: {smart_mean:.1f} ± {smart_std:.1f}",
    f"Stupid: {stupid_mean:.1f} ± {stupid_std:.1f}",
    f"Δ (smart − stupid) = {delta_mean:.1f}",
]
summary_text = "\n".join(summary_lines)

fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
fig.subplots_adjust(bottom=0.22)

ax1 = axes[0]
subjects = sorted(df["subject"].unique())
palette = sns.color_palette("husl", len(subjects))
subject_to_color = {s: palette[i] for i, s in enumerate(subjects)}
colors = [subject_to_color[s] for s in df["subject"]]
ax1.scatter(
    df["stupid_score"],
    df["smart_score"],
    c=colors,
    s=df["level"] * 15 + 20,
    alpha=0.7,
    edgecolors="white",
    linewidths=0.5,
)
ax1.plot([0, 100], [0, 100], "k--", alpha=0.4)
ax1.set_xlabel("Stupid Score")
ax1.set_ylabel("Smart Score")
ax1.set_title("Smart vs Stupid")
ax1.set_xlim(-5, 105)
ax1.set_ylim(-5, 105)
ax1.set_aspect("equal")
subject_handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=palette[i], markersize=8, label=s) for i, s in enumerate(subjects)]
size_handles = [plt.scatter([], [], s=lev * 15 + 20, c="gray", alpha=0.7, label=f"L{lev}") for lev in sorted(df["level"].unique())]
ax1.legend(handles=subject_handles + size_handles + [plt.Line2D([0], [0], color="k", linestyle="--", label="y=x")], loc="lower right", ncol=2)

ax2 = axes[1]
sns.boxplot(data=df_long, x="level", y="score", hue="model", ax=ax2, palette={"Smart": "#2ecc71", "Stupid": "#e74c3c"})
ax2.set_xlabel("Level")
ax2.set_ylabel("Score")
ax2.set_title("Score by Level")
ax2.set_ylim(-5, 105)
ax2.legend(title="Model")

ax3 = axes[2]
sns.boxplot(data=df_long, x="subject", y="score", hue="model", ax=ax3, palette={"Smart": "#2ecc71", "Stupid": "#e74c3c"})
ax3.set_xlabel("Subject")
ax3.set_ylabel("Score")
ax3.set_title("Score by Subject")
ax3.set_ylim(-5, 105)
ax3.tick_params(axis="x", rotation=45)
ax3.legend(title="Model")

fig.suptitle("Evaluation: Smart vs Stupid Model Scores", fontsize=14, fontweight="bold", y=1.02)
fig.text(0.5, 0.04, summary_text, transform=fig.transFigure, fontsize=9, verticalalignment="bottom", horizontalalignment="center", family="monospace", bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8))
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.show()
