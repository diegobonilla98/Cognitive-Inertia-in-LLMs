import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_data(
    baseline_path: str,
    stupid_hacked_path: str,
    smart_hacked_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline = pd.read_csv(baseline_path)
    stupid_hacked = pd.read_csv(stupid_hacked_path)
    smart_hacked = pd.read_csv(smart_hacked_path)

    baseline = baseline[["unique_id", "smart_score", "stupid_score"]].copy()

    if "original_stupid_score" not in stupid_hacked.columns:
        stupid_hacked = stupid_hacked.merge(
            baseline[["unique_id", "stupid_score"]], on="unique_id", how="left"
        )
        stupid_hacked["original_stupid_score"] = stupid_hacked["stupid_score"]

    if "original_smart_score" not in smart_hacked.columns:
        smart_hacked = smart_hacked.merge(
            baseline[["unique_id", "smart_score"]], on="unique_id", how="left"
        )
        smart_hacked["original_smart_score"] = smart_hacked["smart_score"]

    stupid_hacked = stupid_hacked[["unique_id", "original_stupid_score", "hacked_stupid_score"]].copy()
    smart_hacked = smart_hacked[["unique_id", "original_smart_score", "hacked_smart_score"]].copy()

    stupid_hacked["delta"] = stupid_hacked["hacked_stupid_score"] - stupid_hacked["original_stupid_score"]
    smart_hacked["delta"] = smart_hacked["hacked_smart_score"] - smart_hacked["original_smart_score"]

    return baseline, stupid_hacked, smart_hacked


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.2, linewidth=0.8)


def build_plot(
    baseline: pd.DataFrame,
    stupid_hacked: pd.DataFrame,
    smart_hacked: pd.DataFrame,
    output_path: str,
    show: bool,
):
    sns.set_theme(style="whitegrid", context="talk")

    colors = {
        "smart": "#1f77b4",
        "stupid": "#e74c3c",
        "improve": "#2ecc71",
        "worse": "#e67e22",
        "neutral": "#7f8c8d",
    }

    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.18)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    baseline_long = baseline.melt(
        id_vars=["unique_id"],
        value_vars=["smart_score", "stupid_score"],
        var_name="model",
        value_name="score",
    )
    baseline_long["model"] = baseline_long["model"].map(
        {"smart_score": "Smart (baseline)", "stupid_score": "Stupid (baseline)"}
    )
    sns.violinplot(
        data=baseline_long,
        x="model",
        y="score",
        hue="model",
        legend=False,
        ax=ax1,
        inner=None,
        linewidth=1.0,
        palette={
            "Smart (baseline)": colors["smart"],
            "Stupid (baseline)": colors["stupid"],
        },
        saturation=0.8,
    )
    sns.boxplot(
        data=baseline_long,
        x="model",
        y="score",
        ax=ax1,
        width=0.22,
        showcaps=True,
        boxprops={"facecolor": "white", "alpha": 0.85},
        whiskerprops={"linewidth": 1.4},
        medianprops={"color": "black", "linewidth": 2},
        flierprops={"marker": ""},
    )
    ax1.set_title("Baseline Score Distributions", fontweight="bold")
    ax1.set_xlabel("")
    ax1.set_ylabel("Score")
    ax1.set_ylim(-2, 102)
    style_axis(ax1)

    before = stupid_hacked["original_stupid_score"]
    after = stupid_hacked["hacked_stupid_score"]
    improved_mask = stupid_hacked["delta"] > 0
    worsened_mask = stupid_hacked["delta"] < 0
    neutral_mask = stupid_hacked["delta"] == 0

    ax2.scatter(before[neutral_mask], after[neutral_mask], s=40, alpha=0.45, color=colors["neutral"], label="No change")
    ax2.scatter(before[improved_mask], after[improved_mask], s=55, alpha=0.75, color=colors["improve"], label="Improved")
    ax2.scatter(before[worsened_mask], after[worsened_mask], s=55, alpha=0.75, color=colors["worse"], label="Worsened")
    ax2.plot([0, 100], [0, 100], linestyle="--", color="black", linewidth=1.2, alpha=0.6)
    ax2.set_xlim(-2, 102)
    ax2.set_ylim(-2, 102)
    ax2.set_xlabel("Original Stupid Score")
    ax2.set_ylabel("Stupid + Smart-History Score")
    ax2.set_title("Stupid Model Shift (Paired by Same Questions)", fontweight="bold")
    style_axis(ax2)
    ax2.legend(frameon=True, fontsize=10, loc="lower right")

    before = smart_hacked["original_smart_score"]
    after = smart_hacked["hacked_smart_score"]
    improved_mask = smart_hacked["delta"] > 0
    worsened_mask = smart_hacked["delta"] < 0
    neutral_mask = smart_hacked["delta"] == 0

    ax3.scatter(before[neutral_mask], after[neutral_mask], s=28, alpha=0.35, color=colors["neutral"], label="No change")
    ax3.scatter(before[improved_mask], after[improved_mask], s=45, alpha=0.75, color=colors["improve"], label="Improved")
    ax3.scatter(before[worsened_mask], after[worsened_mask], s=45, alpha=0.75, color=colors["worse"], label="Worsened")
    ax3.plot([0, 100], [0, 100], linestyle="--", color="black", linewidth=1.2, alpha=0.6)
    ax3.set_xlim(-2, 102)
    ax3.set_ylim(-2, 102)
    ax3.set_xlabel("Original Smart Score")
    ax3.set_ylabel("Smart + Dumb-History Score")
    ax3.set_title("Smart Model Shift (Paired by Same Questions)", fontweight="bold")
    style_axis(ax3)
    ax3.legend(frameon=True, fontsize=10, loc="lower right")

    summary_rows = [
        ("Stupid baseline (all)", baseline["stupid_score"]),
        ("Smart baseline (all)", baseline["smart_score"]),
        ("Stupid targets before", stupid_hacked["original_stupid_score"]),
        ("Stupid targets after", stupid_hacked["hacked_stupid_score"]),
        ("Smart targets before", smart_hacked["original_smart_score"]),
        ("Smart targets after", smart_hacked["hacked_smart_score"]),
    ]
    summary_df = pd.DataFrame(
        [
            {
                "condition": name,
                "mean": values.mean(),
                "sem": values.std(ddof=1) / np.sqrt(len(values)),
            }
            for name, values in summary_rows
        ]
    )
    order = [
        "Stupid baseline (all)",
        "Smart baseline (all)",
        "Stupid targets before",
        "Stupid targets after",
        "Smart targets before",
        "Smart targets after",
    ]
    summary_df["condition"] = pd.Categorical(summary_df["condition"], categories=order, ordered=True)
    summary_df = summary_df.sort_values("condition")

    bar_colors = [
        colors["stupid"],
        colors["smart"],
        "#c0392b",
        "#27ae60",
        "#2980b9",
        "#f39c12",
    ]
    ax4.barh(summary_df["condition"], summary_df["mean"], xerr=summary_df["sem"], color=bar_colors, alpha=0.92)
    ax4.set_xlim(0, 102)
    ax4.set_xlabel("Mean Score (with SEM)")
    ax4.set_title("Overall Experiment Summary", fontweight="bold")
    style_axis(ax4)

    stupid_delta = stupid_hacked["delta"]
    smart_delta = smart_hacked["delta"]
    metrics_text = (
        f"Stupid->Smart history: n={len(stupid_hacked)}, "
        f"mean delta={stupid_delta.mean():+.2f}, "
        f"improved={(stupid_delta > 0).mean() * 100:.1f}%\n"
        f"Smart->Dumb history: n={len(smart_hacked)}, "
        f"mean delta={smart_delta.mean():+.2f}, "
        f"worsened={(smart_delta < 0).mean() * 100:.1f}%\n"
        f"Baseline means: smart={baseline['smart_score'].mean():.2f}, "
        f"stupid={baseline['stupid_score'].mean():.2f}"
    )
    fig.text(
        0.5,
        0.01,
        metrics_text,
        ha="center",
        va="bottom",
        fontsize=12,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#fdf6e3", "edgecolor": "#d6c6a3"},
    )

    fig.suptitle(
        "Context Hacking on MATH-500: Baseline vs Manipulated History",
        fontsize=22,
        fontweight="bold",
        y=0.99,
    )
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Unified evaluation plot for baseline and history-hacking experiments."
    )
    parser.add_argument("--baseline-path", default="results_scores.csv")
    parser.add_argument("--stupid-hacked-path", default="responses_from_stupid_to_smart_scores.csv")
    parser.add_argument("--smart-hacked-path", default="responses_from_smart_to_stupid_scores.csv")
    parser.add_argument("--output-path", default="eval_all_in_one_plot.png")
    parser.add_argument("--show", action="store_true", help="Display the figure window after saving.")
    args = parser.parse_args()

    baseline, stupid_hacked, smart_hacked = load_data(
        baseline_path=args.baseline_path,
        stupid_hacked_path=args.stupid_hacked_path,
        smart_hacked_path=args.smart_hacked_path,
    )
    build_plot(baseline, stupid_hacked, smart_hacked, args.output_path, args.show)


if __name__ == "__main__":
    main()
