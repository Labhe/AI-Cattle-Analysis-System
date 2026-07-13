"""
Model comparison reporting for the AI Cattle Analysis System (Phase 14).

Aggregates metrics from several models/architectures into a comparison table,
a ranked markdown report, and a bar chart. Works for both classification and
regression metric sets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def build_comparison_table(results: Dict[str, Dict[str, Any]],
                           metric_keys: List[str]) -> pd.DataFrame:
    """
    Flatten ``{model_name: metrics_dict}`` into a comparison DataFrame.

    Nested metrics are addressed with dotted keys (e.g. ``macro.f1``);
    missing metrics become NaN.
    """
    rows = []
    for name, metrics in results.items():
        row: Dict[str, Any] = {"model": name}
        for key in metric_keys:
            value: Any = metrics
            for part in key.split("."):
                value = value.get(part) if isinstance(value, dict) else None
                if value is None:
                    break
            row[key] = value
        rows.append(row)
    return pd.DataFrame(rows)


def rank_models(table: pd.DataFrame, metric: str, higher_is_better: bool = True) -> pd.DataFrame:
    """Return the table sorted by ``metric`` with a ``rank`` column."""
    if metric not in table.columns:
        return table
    ranked = table.sort_values(metric, ascending=not higher_is_better).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked


def plot_comparison(table: pd.DataFrame, metric: str, out_path: Path) -> Optional[Path]:
    """Bar chart of one metric across models."""
    if metric not in table.columns:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(max(6, len(table) * 1.2), 5))
    plt.bar(table["model"].astype(str), pd.to_numeric(table[metric], errors="coerce"),
            color="#3b82f6")
    plt.ylabel(metric)
    plt.title(f"Model comparison — {metric}")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    return Path(out_path)


def write_markdown_report(table: pd.DataFrame, out_path: Path,
                          title: str = "Model Comparison Report") -> Path:
    """Write a markdown table report to ``out_path``."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    display = table.copy()
    for col in display.columns:
        if display[col].dtype.kind == "f":
            display[col] = display[col].map(lambda v: f"{v:.4f}" if pd.notna(v) else "—")
    lines.append("| " + " | ".join(display.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(display.columns)) + " |")
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    out_path = Path(out_path)
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def compare_models(
    results: Dict[str, Dict[str, Any]],
    metric_keys: List[str],
    primary_metric: str,
    output_dir: Path,
    higher_is_better: bool = True,
    prefix: str = "comparison",
) -> Dict[str, Any]:
    """
    Produce a full model-comparison report (CSV, markdown, bar chart).

    Returns a dict with the ranked records, the best model name, and the
    written file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = build_comparison_table(results, metric_keys)
    ranked = rank_models(table, primary_metric, higher_is_better)

    csv_path = output_dir / f"{prefix}.csv"
    ranked.to_csv(csv_path, index=False)
    md_path = write_markdown_report(ranked, output_dir / f"{prefix}.md")
    chart = plot_comparison(ranked, primary_metric, output_dir / f"{prefix}_{primary_metric.replace('.', '_')}.png")

    best = ranked.iloc[0]["model"] if len(ranked) and primary_metric in ranked.columns else None
    return {
        "ranked": ranked.to_dict(orient="records"),
        "best_model": best,
        "paths": {"csv": str(csv_path), "markdown": str(md_path),
                  "chart": str(chart) if chart else None},
    }
