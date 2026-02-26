import argparse
import json
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd


def _require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().split())


def _rate(count: int, n: int) -> float:
    if n == 0:
        return 0.0
    return 100.0 * count / n


def _summary_stats(values: pd.Series) -> dict:
    vals = values.astype(float)
    return {
        "n": int(vals.shape[0]),
        "mean": float(vals.mean()),
        "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
        "median": float(vals.median()),
        "min": float(vals.min()),
        "p25": float(vals.quantile(0.25)),
        "p75": float(vals.quantile(0.75)),
        "max": float(vals.max()),
    }


def _delta_stats(delta: pd.Series) -> dict:
    vals = delta.astype(float)
    n = int(vals.shape[0])
    mean = float(vals.mean())
    std = float(vals.std(ddof=1)) if n > 1 else 0.0
    se = std / math.sqrt(n) if n > 0 and std > 0 else 0.0
    ci_low = mean - 1.96 * se
    ci_high = mean + 1.96 * se
    improved = int((vals > 0).sum())
    worsened = int((vals < 0).sum())
    unchanged = int((vals == 0).sum())
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "median": float(vals.median()),
        "min": float(vals.min()),
        "p10": float(vals.quantile(0.10)),
        "p25": float(vals.quantile(0.25)),
        "p75": float(vals.quantile(0.75)),
        "p90": float(vals.quantile(0.90)),
        "max": float(vals.max()),
        "improved_count": improved,
        "worsened_count": worsened,
        "unchanged_count": unchanged,
        "improved_rate": _rate(improved, n),
        "worsened_rate": _rate(worsened, n),
        "unchanged_rate": _rate(unchanged, n),
        "se": se,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }


def _binomial_pmf_half(n: int) -> list[float]:
    if n < 0:
        return []
    if n == 0:
        return [1.0]
    pmf = [0.0] * (n + 1)
    pmf[0] = 0.5 ** n
    for k in range(0, n):
        pmf[k + 1] = pmf[k] * (n - k) / (k + 1)
    return pmf


def _sign_test_two_sided(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = wins
    pmf = _binomial_pmf_half(n)
    p_obs = pmf[k]
    return float(sum(p for p in pmf if p <= p_obs + 1e-15))


def _markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def _f(value: float, digits: int = 2) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float) and math.isnan(value):
        return "NA"
    return f"{value:.{digits}f}"


def _as_records(df: pd.DataFrame) -> list[dict]:
    records = []
    for record in df.to_dict(orient="records"):
        normalized = {}
        for key, value in record.items():
            if isinstance(value, (np.floating, float)):
                normalized[key] = float(value)
            elif isinstance(value, (np.integer, int)):
                normalized[key] = int(value)
            else:
                normalized[key] = value
        records.append(normalized)
    return records


def _intervention_metrics(
    df: pd.DataFrame,
    before_score_col: str,
    after_score_col: str,
    original_response_col: str,
    hacked_response_col: str,
) -> dict:
    working = df.copy()
    working["delta"] = working[after_score_col] - working[before_score_col]
    working["same_response"] = (
        working[original_response_col].map(_normalize_text)
        == working[hacked_response_col].map(_normalize_text)
    )

    before_std = float(working[before_score_col].astype(float).std(ddof=1)) if len(working) > 1 else 0.0
    if before_std == 0.0:
        delta_corr = float("nan")
    else:
        delta_corr = float(working["delta"].corr(working[before_score_col]))

    summary = {
        "n": int(len(working)),
        "before": _summary_stats(working[before_score_col]),
        "after": _summary_stats(working[after_score_col]),
        "delta": _delta_stats(working["delta"]),
        "same_response_count": int(working["same_response"].sum()),
        "same_response_rate": _rate(int(working["same_response"].sum()), len(working)),
        "changed_response_count": int((~working["same_response"]).sum()),
        "changed_response_rate": _rate(int((~working["same_response"]).sum()), len(working)),
        "same_response_delta_mean": float(
            working.loc[working["same_response"], "delta"].mean()
            if (working["same_response"]).any()
            else 0.0
        ),
        "changed_response_delta_mean": float(
            working.loc[~working["same_response"], "delta"].mean()
            if (~working["same_response"]).any()
            else 0.0
        ),
        "same_response_improved_count": int(((working["same_response"]) & (working["delta"] > 0)).sum()),
        "same_response_worsened_count": int(((working["same_response"]) & (working["delta"] < 0)).sum()),
        "changed_response_improved_count": int(((~working["same_response"]) & (working["delta"] > 0)).sum()),
        "changed_response_worsened_count": int(((~working["same_response"]) & (working["delta"] < 0)).sum()),
        "score_thresholds": {
            "ge_80_before": int((working[before_score_col] >= 80).sum()),
            "ge_80_after": int((working[after_score_col] >= 80).sum()),
            "ge_90_before": int((working[before_score_col] >= 90).sum()),
            "ge_90_after": int((working[after_score_col] >= 90).sum()),
            "ge_95_before": int((working[before_score_col] >= 95).sum()),
            "ge_95_after": int((working[after_score_col] >= 95).sum()),
            "eq_100_before": int((working[before_score_col] == 100).sum()),
            "eq_100_after": int((working[after_score_col] == 100).sum()),
        },
        "delta_corr_with_before": delta_corr,
    }

    wins = summary["delta"]["improved_count"]
    losses = summary["delta"]["worsened_count"]
    summary["sign_test_two_sided_p"] = _sign_test_two_sided(wins=wins, losses=losses)

    by_subject = (
        working.groupby("subject")
        .agg(
            n=("unique_id", "count"),
            before_mean=(before_score_col, "mean"),
            after_mean=(after_score_col, "mean"),
            delta_mean=("delta", "mean"),
            improve_rate=("delta", lambda s: (s > 0).mean() * 100.0),
            worsen_rate=("delta", lambda s: (s < 0).mean() * 100.0),
            same_response_rate=("same_response", lambda s: s.mean() * 100.0),
        )
        .sort_values("delta_mean", ascending=False)
        .reset_index()
    )
    summary["by_subject"] = _as_records(by_subject)

    by_level = (
        working.groupby("level")
        .agg(
            n=("unique_id", "count"),
            before_mean=(before_score_col, "mean"),
            after_mean=(after_score_col, "mean"),
            delta_mean=("delta", "mean"),
            improve_rate=("delta", lambda s: (s > 0).mean() * 100.0),
            worsen_rate=("delta", lambda s: (s < 0).mean() * 100.0),
            same_response_rate=("same_response", lambda s: s.mean() * 100.0),
        )
        .sort_values("level")
        .reset_index()
    )
    summary["by_level"] = _as_records(by_level)

    score_band = pd.cut(
        working[before_score_col],
        bins=[-1, 19, 39, 59, 79, 99, 100],
        labels=["0-19", "20-39", "40-59", "60-79", "80-99", "100"],
        include_lowest=True,
    )
    by_band = (
        working.assign(start_band=score_band)
        .groupby("start_band", observed=False)
        .agg(
            n=("unique_id", "count"),
            before_mean=(before_score_col, "mean"),
            after_mean=(after_score_col, "mean"),
            delta_mean=("delta", "mean"),
            improve_rate=("delta", lambda s: (s > 0).mean() * 100.0),
            worsen_rate=("delta", lambda s: (s < 0).mean() * 100.0),
        )
        .reset_index()
    )
    summary["by_start_band"] = _as_records(by_band)

    top_gain = (
        working[["unique_id", "subject", "level", before_score_col, after_score_col, "delta"]]
        .sort_values("delta", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    top_drop = (
        working[["unique_id", "subject", "level", before_score_col, after_score_col, "delta"]]
        .sort_values("delta", ascending=True)
        .head(10)
        .reset_index(drop=True)
    )
    summary["top_gains"] = _as_records(top_gain)
    summary["top_drops"] = _as_records(top_drop)

    return summary


def compute_metrics(
    baseline_path: str,
    stupid_to_smart_path: str,
    smart_to_stupid_path: str,
) -> dict:
    baseline = pd.read_csv(baseline_path)
    s2s = pd.read_csv(stupid_to_smart_path)
    sm2st = pd.read_csv(smart_to_stupid_path)

    _require_columns(
        baseline,
        ["unique_id", "smart_score", "stupid_score", "subject", "level"],
        "baseline csv",
    )
    _require_columns(
        s2s,
        [
            "unique_id",
            "hacked_stupid_score",
            "original_stupid_score",
            "original_stupid_response",
            "hacked_stupid_response",
            "subject",
            "level",
        ],
        "stupid->smart csv",
    )
    _require_columns(
        sm2st,
        [
            "unique_id",
            "hacked_smart_score",
            "original_smart_score",
            "original_smart_response",
            "hacked_smart_response",
            "subject",
            "level",
        ],
        "smart->stupid csv",
    )

    baseline_summary = {
        "n": int(len(baseline)),
        "smart": _summary_stats(baseline["smart_score"]),
        "stupid": _summary_stats(baseline["stupid_score"]),
        "gap_smart_minus_stupid": _summary_stats(baseline["smart_score"] - baseline["stupid_score"]),
        "smart_perfect_count": int((baseline["smart_score"] == 100).sum()),
        "stupid_perfect_count": int((baseline["stupid_score"] == 100).sum()),
    }

    s2s_summary = _intervention_metrics(
        s2s,
        before_score_col="original_stupid_score",
        after_score_col="hacked_stupid_score",
        original_response_col="original_stupid_response",
        hacked_response_col="hacked_stupid_response",
    )
    sm2st_summary = _intervention_metrics(
        sm2st,
        before_score_col="original_smart_score",
        after_score_col="hacked_smart_score",
        original_response_col="original_smart_response",
        hacked_response_col="hacked_smart_response",
    )

    s2s_vs_base = (
        s2s[["unique_id", "hacked_stupid_score"]]
        .merge(
            baseline[["unique_id", "smart_score", "stupid_score"]],
            on="unique_id",
            how="left",
        )
        .copy()
    )
    s2s_same_question_summary = {
        "n": int(len(s2s_vs_base)),
        "hacked_stupid_mean": float(s2s_vs_base["hacked_stupid_score"].mean()),
        "baseline_smart_mean_same_questions": float(s2s_vs_base["smart_score"].mean()),
        "baseline_stupid_mean_same_questions": float(s2s_vs_base["stupid_score"].mean()),
        "hacked_ge_baseline_smart_count": int(
            (s2s_vs_base["hacked_stupid_score"] >= s2s_vs_base["smart_score"]).sum()
        ),
        "hacked_ge_baseline_smart_rate": _rate(
            int((s2s_vs_base["hacked_stupid_score"] >= s2s_vs_base["smart_score"]).sum()),
            len(s2s_vs_base),
        ),
        "hacked_gt_baseline_stupid_count": int(
            (s2s_vs_base["hacked_stupid_score"] > s2s_vs_base["stupid_score"]).sum()
        ),
        "hacked_gt_baseline_stupid_rate": _rate(
            int((s2s_vs_base["hacked_stupid_score"] > s2s_vs_base["stupid_score"]).sum()),
            len(s2s_vs_base),
        ),
    }

    overall = baseline[["unique_id", "smart_score", "stupid_score"]].copy()
    overall = overall.merge(
        s2s[["unique_id", "hacked_stupid_score"]],
        on="unique_id",
        how="left",
    )
    overall = overall.merge(
        sm2st[["unique_id", "hacked_smart_score"]],
        on="unique_id",
        how="left",
    )
    overall["stupid_experiment_score"] = overall["hacked_stupid_score"].fillna(overall["stupid_score"])
    overall["smart_experiment_score"] = overall["hacked_smart_score"].fillna(overall["smart_score"])

    overall_summary = {
        "n": int(len(overall)),
        "stupid_baseline_mean": float(overall["stupid_score"].mean()),
        "stupid_experiment_mean": float(overall["stupid_experiment_score"].mean()),
        "stupid_mean_delta": float((overall["stupid_experiment_score"] - overall["stupid_score"]).mean()),
        "smart_baseline_mean": float(overall["smart_score"].mean()),
        "smart_experiment_mean": float(overall["smart_experiment_score"].mean()),
        "smart_mean_delta": float((overall["smart_experiment_score"] - overall["smart_score"]).mean()),
    }

    large_effects = {
        "s2s_large_gains": {
            f"ge_{threshold}": int((s2s["hacked_stupid_score"] - s2s["original_stupid_score"] >= threshold).sum())
            for threshold in [1, 3, 5, 10, 20, 30]
        },
        "s2s_large_drops": {
            f"le_neg_{threshold}": int((s2s["hacked_stupid_score"] - s2s["original_stupid_score"] <= -threshold).sum())
            for threshold in [1, 3, 5, 10, 20, 30]
        },
        "sm2st_large_drops": {
            f"le_neg_{threshold}": int((sm2st["hacked_smart_score"] - sm2st["original_smart_score"] <= -threshold).sum())
            for threshold in [1, 3, 5, 10, 20, 30]
        },
    }

    return {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "source_files": {
            "baseline": baseline_path,
            "stupid_to_smart_scores": stupid_to_smart_path,
            "smart_to_stupid_scores": smart_to_stupid_path,
        },
        "baseline": baseline_summary,
        "stupid_to_smart": s2s_summary,
        "smart_to_stupid": sm2st_summary,
        "same_question_comparison_stupid_to_smart_targets": s2s_same_question_summary,
        "overall_experiment": overall_summary,
        "large_effect_counts": large_effects,
    }


def _subject_rows(records: list[dict]) -> list[list[object]]:
    rows = []
    for r in records:
        rows.append(
            [
                r["subject"],
                int(r["n"]),
                _f(r["before_mean"]),
                _f(r["after_mean"]),
                _f(r["delta_mean"]),
                f"{_f(r['improve_rate'])}%",
                f"{_f(r['worsen_rate'])}%",
                f"{_f(r['same_response_rate'])}%",
            ]
        )
    return rows


def _level_rows(records: list[dict]) -> list[list[object]]:
    rows = []
    for r in records:
        rows.append(
            [
                int(r["level"]),
                int(r["n"]),
                _f(r["before_mean"]),
                _f(r["after_mean"]),
                _f(r["delta_mean"]),
                f"{_f(r['improve_rate'])}%",
                f"{_f(r['worsen_rate'])}%",
            ]
        )
    return rows


def _band_rows(records: list[dict]) -> list[list[object]]:
    rows = []
    for r in records:
        rows.append(
            [
                r["start_band"],
                int(r["n"]),
                _f(r["before_mean"]) if not pd.isna(r["before_mean"]) else "NA",
                _f(r["after_mean"]) if not pd.isna(r["after_mean"]) else "NA",
                _f(r["delta_mean"]) if not pd.isna(r["delta_mean"]) else "NA",
                f"{_f(r['improve_rate'])}%" if not pd.isna(r["improve_rate"]) else "NA",
                f"{_f(r['worsen_rate'])}%" if not pd.isna(r["worsen_rate"]) else "NA",
            ]
        )
    return rows


def _top_rows(records: list[dict], before_col: str, after_col: str) -> list[list[object]]:
    rows = []
    for r in records:
        rows.append(
            [
                int(r["unique_id"]),
                r["subject"],
                int(r["level"]),
                int(r[before_col]),
                int(r[after_col]),
                int(r["delta"]),
            ]
        )
    return rows


def render_report(metrics: dict) -> str:
    base = metrics["baseline"]
    s2s = metrics["stupid_to_smart"]
    sm2st = metrics["smart_to_stupid"]
    same_q = metrics["same_question_comparison_stupid_to_smart_targets"]
    overall = metrics["overall_experiment"]
    large = metrics["large_effect_counts"]

    s2s_delta = s2s["delta"]
    sm2st_delta = sm2st["delta"]
    s2s_thr = s2s["score_thresholds"]
    sm2st_thr = sm2st["score_thresholds"]

    s2s_subject_table = _markdown_table(
        ["Subject", "n", "Before", "After", "Delta", "Improve", "Worsen", "Same response"],
        _subject_rows(s2s["by_subject"]),
    )
    sm2st_subject_table = _markdown_table(
        ["Subject", "n", "Before", "After", "Delta", "Improve", "Worsen", "Same response"],
        _subject_rows(sm2st["by_subject"]),
    )
    s2s_level_table = _markdown_table(
        ["Level", "n", "Before", "After", "Delta", "Improve", "Worsen"],
        _level_rows(s2s["by_level"]),
    )
    sm2st_level_table = _markdown_table(
        ["Level", "n", "Before", "After", "Delta", "Improve", "Worsen"],
        _level_rows(sm2st["by_level"]),
    )
    s2s_band_table = _markdown_table(
        ["Start band", "n", "Before", "After", "Delta", "Improve", "Worsen"],
        _band_rows(s2s["by_start_band"]),
    )
    sm2st_band_table = _markdown_table(
        ["Start band", "n", "Before", "After", "Delta", "Improve", "Worsen"],
        _band_rows(sm2st["by_start_band"]),
    )

    s2s_top_gain_table = _markdown_table(
        ["unique_id", "Subject", "Level", "Before", "After", "Delta"],
        _top_rows(s2s["top_gains"], "original_stupid_score", "hacked_stupid_score"),
    )
    s2s_top_drop_table = _markdown_table(
        ["unique_id", "Subject", "Level", "Before", "After", "Delta"],
        _top_rows(s2s["top_drops"], "original_stupid_score", "hacked_stupid_score"),
    )
    sm2st_top_drop_table = _markdown_table(
        ["unique_id", "Subject", "Level", "Before", "After", "Delta"],
        _top_rows(sm2st["top_drops"], "original_smart_score", "hacked_smart_score"),
    )

    return f"""# Context Hacking on MATH-500: Mega Study Report

Date: {datetime.now().strftime("%Y-%m-%d")}  
Project: `ContextMatters`  
Report source: auto-generated by `measure_study_results.py` from CSV outputs.

## Executive Summary

This study tested if fake conversation history can steer model quality on MATH-500:

1. Improve the weak model by prepending strong-model perfect examples from the same subject.
2. Degrade the strong model by prepending weak-model low-score examples from the same subject.

Headline results:

- Weak model on failure set (`n={s2s["n"]}`): mean score `{_f(s2s["before"]["mean"])} -> {_f(s2s["after"]["mean"])}` (`{_f(s2s_delta["mean"])} delta`).
- Strong model on perfect set (`n={sm2st["n"]}`): mean score `{_f(sm2st["before"]["mean"])} -> {_f(sm2st["after"]["mean"])}` (`{_f(sm2st_delta["mean"])} delta`).
- Net whole-dataset impact (`n={overall["n"]}`):
  - Weak track: `{_f(overall["stupid_baseline_mean"])} -> {_f(overall["stupid_experiment_mean"])}` (`{_f(overall["stupid_mean_delta"])} delta`)
  - Strong track: `{_f(overall["smart_baseline_mean"])} -> {_f(overall["smart_experiment_mean"])}` (`{_f(overall["smart_mean_delta"])} delta`)

## Experiment Context

- Baseline file: `{metrics["source_files"]["baseline"]}`
- Weak-history intervention scores: `{metrics["source_files"]["stupid_to_smart_scores"]}`
- Strong-history intervention scores: `{metrics["source_files"]["smart_to_stupid_scores"]}`
- Baseline answer/scoring used:
  - Smart model: `gpt-5.2-2025-12-11`
  - Weak model: `gpt-4o-mini-2024-07-18`
  - Grader: smart model (`gpt-5.2`) with 0-100 integer rubric prompt

## Baseline Metrics

| Metric | Smart | Weak |
|---|---:|---:|
| N | {base["n"]} | {base["n"]} |
| Mean | {_f(base["smart"]["mean"])} | {_f(base["stupid"]["mean"])} |
| Std | {_f(base["smart"]["std"])} | {_f(base["stupid"]["std"])} |
| Median | {_f(base["smart"]["median"])} | {_f(base["stupid"]["median"])} |
| P25 | {_f(base["smart"]["p25"])} | {_f(base["stupid"]["p25"])} |
| P75 | {_f(base["smart"]["p75"])} | {_f(base["stupid"]["p75"])} |
| Perfect (100) | {base["smart_perfect_count"]} | {base["stupid_perfect_count"]} |

Baseline gap (smart minus weak):

- Mean: `{_f(base["gap_smart_minus_stupid"]["mean"])}`
- Median: `{_f(base["gap_smart_minus_stupid"]["median"])}`
- Std: `{_f(base["gap_smart_minus_stupid"]["std"])}`

## Intervention A: Weak Model with Smart History

### Core paired effects (`n={s2s["n"]}`)

| Metric | Value |
|---|---:|
| Mean before | {_f(s2s["before"]["mean"])} |
| Mean after | {_f(s2s["after"]["mean"])} |
| Mean delta | {_f(s2s_delta["mean"])} |
| Median delta | {_f(s2s_delta["median"])} |
| Delta std | {_f(s2s_delta["std"])} |
| Delta 95% CI | [{_f(s2s_delta["ci95_low"])}, {_f(s2s_delta["ci95_high"])}] |
| Improved | {s2s_delta["improved_count"]} ({_f(s2s_delta["improved_rate"])}%) |
| Worsened | {s2s_delta["worsened_count"]} ({_f(s2s_delta["worsened_rate"])}%) |
| Unchanged | {s2s_delta["unchanged_count"]} ({_f(s2s_delta["unchanged_rate"])}%) |
| Sign test p (two-sided, no ties) | {_f(s2s["sign_test_two_sided_p"], 4)} |

Threshold movement:

- `>=80`: {s2s_thr["ge_80_before"]} -> {s2s_thr["ge_80_after"]}
- `>=90`: {s2s_thr["ge_90_before"]} -> {s2s_thr["ge_90_after"]}
- `>=95`: {s2s_thr["ge_95_before"]} -> {s2s_thr["ge_95_after"]}
- `=100`: {s2s_thr["eq_100_before"]} -> {s2s_thr["eq_100_after"]}

Same-response behavior:

- Exact same response text: {s2s["same_response_count"]}/{s2s["n"]} ({_f(s2s["same_response_rate"])}%)
- Changed response text: {s2s["changed_response_count"]}/{s2s["n"]} ({_f(s2s["changed_response_rate"])}%)
- Mean delta when response unchanged: {_f(s2s["same_response_delta_mean"])}
- Mean delta when response changed: {_f(s2s["changed_response_delta_mean"])}
- Delta correlation with starting score: {_f(s2s["delta_corr_with_before"])}

By subject:

{s2s_subject_table}

By level:

{s2s_level_table}

By starting score band:

{s2s_band_table}

Top gains:

{s2s_top_gain_table}

Top drops:

{s2s_top_drop_table}

## Intervention B: Strong Model with Weak History

### Core paired effects (`n={sm2st["n"]}`)

| Metric | Value |
|---|---:|
| Mean before | {_f(sm2st["before"]["mean"])} |
| Mean after | {_f(sm2st["after"]["mean"])} |
| Mean delta | {_f(sm2st_delta["mean"])} |
| Median delta | {_f(sm2st_delta["median"])} |
| Delta std | {_f(sm2st_delta["std"])} |
| Delta 95% CI | [{_f(sm2st_delta["ci95_low"])}, {_f(sm2st_delta["ci95_high"])}] |
| Improved | {sm2st_delta["improved_count"]} ({_f(sm2st_delta["improved_rate"])}%) |
| Worsened | {sm2st_delta["worsened_count"]} ({_f(sm2st_delta["worsened_rate"])}%) |
| Unchanged | {sm2st_delta["unchanged_count"]} ({_f(sm2st_delta["unchanged_rate"])}%) |
| Sign test p (two-sided, no ties) | {_f(sm2st["sign_test_two_sided_p"], 4)} |

Threshold movement:

- `>=80`: {sm2st_thr["ge_80_before"]} -> {sm2st_thr["ge_80_after"]}
- `>=90`: {sm2st_thr["ge_90_before"]} -> {sm2st_thr["ge_90_after"]}
- `>=95`: {sm2st_thr["ge_95_before"]} -> {sm2st_thr["ge_95_after"]}
- `=100`: {sm2st_thr["eq_100_before"]} -> {sm2st_thr["eq_100_after"]}

Same-response behavior:

- Exact same response text: {sm2st["same_response_count"]}/{sm2st["n"]} ({_f(sm2st["same_response_rate"])}%)
- Changed response text: {sm2st["changed_response_count"]}/{sm2st["n"]} ({_f(sm2st["changed_response_rate"])}%)
- Mean delta when response unchanged: {_f(sm2st["same_response_delta_mean"])}
- Mean delta when response changed: {_f(sm2st["changed_response_delta_mean"])}
- Delta correlation with starting score: {_f(sm2st["delta_corr_with_before"])}

By subject:

{sm2st_subject_table}

By level:

{sm2st_level_table}

By starting score band:

{sm2st_band_table}

Largest drops:

{sm2st_top_drop_table}

## Same-question contrast for weak intervention set

On the exact 154 target questions used in weak intervention:

- Hacked weak mean: {_f(same_q["hacked_stupid_mean"])}
- Baseline smart mean (same questions): {_f(same_q["baseline_smart_mean_same_questions"])}
- Baseline weak mean (same questions): {_f(same_q["baseline_stupid_mean_same_questions"])}
- Hacked weak >= baseline smart: {same_q["hacked_ge_baseline_smart_count"]}/{same_q["n"]} ({_f(same_q["hacked_ge_baseline_smart_rate"])}%)
- Hacked weak > baseline weak: {same_q["hacked_gt_baseline_stupid_count"]}/{same_q["n"]} ({_f(same_q["hacked_gt_baseline_stupid_rate"])}%)

## Overall experiment impact

| Track | Baseline mean | Post-intervention mean | Mean delta |
|---|---:|---:|---:|
| Weak track | {_f(overall["stupid_baseline_mean"])} | {_f(overall["stupid_experiment_mean"])} | {_f(overall["stupid_mean_delta"])} |
| Strong track | {_f(overall["smart_baseline_mean"])} | {_f(overall["smart_experiment_mean"])} | {_f(overall["smart_mean_delta"])} |

Large-effect counts:

- Weak gains >= 10: {large["s2s_large_gains"]["ge_10"]}, >=20: {large["s2s_large_gains"]["ge_20"]}, >=30: {large["s2s_large_gains"]["ge_30"]}
- Weak drops <= -10: {large["s2s_large_drops"]["le_neg_10"]}, <=-20: {large["s2s_large_drops"]["le_neg_20"]}, <=-30: {large["s2s_large_drops"]["le_neg_30"]}
- Strong drops <= -1: {large["sm2st_large_drops"]["le_neg_1"]}, <=-5: {large["sm2st_large_drops"]["le_neg_5"]}, <=-10: {large["sm2st_large_drops"]["le_neg_10"]}

## Conclusions

1. Smart-history injection improves weak model performance on average, but with high variance and meaningful downside risk.
2. Improvements are concentrated on low/mid weak-baseline items; higher weak-baseline items are vulnerable to regression.
3. Strong model behavior is highly resistant to this dumb-history corruption strategy; degradation is sparse and mostly small.
4. A practical deployment should use conditional gating before applying history injection.

## Threats to validity

1. The scorer is the same smart model family used in generation, which can induce grading bias.
2. Target sets are asymmetric (`weak <100` versus `smart =100`), so intervention comparisons are not apples-to-apples.
3. Single-run sampling per question leaves stochastic variance under-measured.
4. Near-ceiling baseline scores compress measurable degradation for the strong model.

---

Generated at: {metrics["generated_at_utc"]}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure and report context-hacking study results.")
    parser.add_argument("--baseline-path", default="results_scores.csv")
    parser.add_argument("--stupid-to-smart-path", default="responses_from_stupid_to_smart_scores.csv")
    parser.add_argument("--smart-to-stupid-path", default="responses_from_smart_to_stupid_scores.csv")
    parser.add_argument("--json-output", default="study_metrics.json")
    parser.add_argument("--report-output", default="MEGA_STUDY_RESULTS.md")
    args = parser.parse_args()

    metrics = compute_metrics(
        baseline_path=args.baseline_path,
        stupid_to_smart_path=args.stupid_to_smart_path,
        smart_to_stupid_path=args.smart_to_stupid_path,
    )

    with open(args.json_output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    report = render_report(metrics)
    with open(args.report_output, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Wrote metrics json: {args.json_output}")
    print(f"Wrote report markdown: {args.report_output}")


if __name__ == "__main__":
    main()
