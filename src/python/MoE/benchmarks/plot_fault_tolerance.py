#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = Path(__file__).resolve().parent / "data" / "fault_tolerance" / "rdma" / "nixl"
NIXL_CSV = DATA_DIR / "fault_tolerance_rps_timeline_rps_50.0.csv"
DEEPEP_CSV = DATA_DIR / "fault_tolerance_rps_timeline_rps_50.0_deepep.csv"
SOURCE_PNG = DATA_DIR / "fault_tolerance_rps_timeline_rps_50.0.png"
DOCS_DIR = ROOT / "docs" / "diploma" / "bachelor-thesis-template" / "diagrams" / "experiments"
DOCS_PNG = DOCS_DIR / "fault_tolerance_rps_timeline_rps_50.png"
DOCS_PDF = DOCS_DIR / "fault_tolerance_rps_timeline_rps_50.pdf"


def load_timeline(path: Path) -> pd.DataFrame:
    timeline = pd.read_csv(path)
    required_columns = {"time_s", "rps", "p01", "p99"}
    missing_columns = required_columns - set(timeline.columns)
    if missing_columns:
        raise ValueError(f"{path} does not contain columns: {sorted(missing_columns)}")
    if (timeline["p01"] > timeline["rps"]).any() or (timeline["rps"] > timeline["p99"]).any():
        raise ValueError(f"{path} contains an RPS value outside its P01-P99 interval")
    return timeline


def add_series(
    ax: plt.Axes,
    timeline: pd.DataFrame,
    *,
    color: str,
    marker: str,
    label: str,
) -> None:
    time = timeline["time_s"]
    ax.fill_between(
        time,
        timeline["p01"],
        timeline["p99"],
        color=color,
        alpha=0.12,
        linewidth=0,
        zorder=1,
    )
    ax.plot(
        time,
        timeline["rps"],
        color=color,
        marker=marker,
        linewidth=2.4,
        markersize=4.5,
        label=f"Observed RPS ({label})",
        zorder=3,
    )


def build_plot() -> plt.Figure:
    nixl = load_timeline(NIXL_CSV)
    deepep = load_timeline(DEEPEP_CSV)

    fig, ax = plt.subplots(figsize=(16, 6))
    add_series(ax, nixl, color="#1f77b4", marker="o", label="NIXL")
    add_series(ax, deepep, color="#111111", marker="s", label="DeepEP")

    ax.axhline(50, color="#9467bd", linestyle=":", linewidth=2, label="Target RPS")
    ax.axvline(10, color="#d62728", linestyle="--", linewidth=2, label="Expert block failure")
    ax.axvline(20, color="#2ca02c", linestyle="--", linewidth=2, label="Expert block restore")

    ax.set_xlim(0, 35)
    ax.set_ylim(-2, 52.5)
    ax.set_xlabel("Time, s", fontsize=12)
    ax.set_ylabel("Requests per second", fontsize=12)
    ax.set_title("Fault tolerance benchmark, target 50 RPS", fontsize=15)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", fontsize=9.5, frameon=True, ncol=2)
    fig.tight_layout()
    return fig


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    fig = build_plot()
    fig.savefig(SOURCE_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(DOCS_PNG, dpi=180, bbox_inches="tight")
    fig.savefig(DOCS_PDF, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
