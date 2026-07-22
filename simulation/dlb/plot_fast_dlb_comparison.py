#!/usr/bin/env python3
"""Render FAST and DLB comparison figures from the simulation benchmark CSV."""

import csv
import sys
from pathlib import Path


DLB_COLOR = "#2A9D8F"
FAST_COLOR = "#457B9D"
FAST_STAGE_COLOR = "#E76F51"
FONT_SIZE = 20
LEGEND_FONT_SIZE = 14
ANNOTATION_FONT_SIZE = 12


def get_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": LEGEND_FONT_SIZE,
        }
    )
    import matplotlib.pyplot as plt

    return plt


def load_rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as handle:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def annotate_bars(axis, bars, values) -> None:
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.0f}",
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_FONT_SIZE,
        )


def plot_scheduler_and_bottleneck(rows: list[dict[str, float]], output_dir: Path) -> None:
    plt = get_pyplot()
    servers = [int(row["servers"]) for row in rows]

    figure, axes = plt.subplots(2, 1, figsize=(10, 8))
    figure.subplots_adjust(left=0.17, right=0.98, bottom=0.10, top=0.93, hspace=0.95)
    axes[0].plot(servers, [row["dlb_planner_us"] for row in rows], marker="o", linewidth=2,
                 color=DLB_COLOR, label="DLB local")
    axes[0].plot(servers, [row["fast_planner_us"] for row in rows], marker="s", linewidth=2,
                 color=FAST_COLOR, label="FAST + Birkhoff")
    axes[0].set_title("Host Planner Time")
    axes[0].set_xlabel("Servers (M = 8 GPUs/NICs)")
    axes[0].set_ylabel("Planner time (us)")
    axes[0].set_xticks(servers)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    axes[1].plot(servers, [row["fast_scaleout_bottleneck_records"] for row in rows], marker="s", linewidth=2,
                 color=FAST_COLOR, label="FAST Rail lower bound")
    axes[1].plot(servers, [row["dlb_scaleout_bottleneck_records"] for row in rows], marker="o", linewidth=2,
                 linestyle="--", markerfacecolor="none", markeredgewidth=2.5,
                 color=DLB_COLOR, label="DLB NIC bottleneck")
    axes[1].set_title("Scale-Out Bottleneck Volume")
    axes[1].set_xlabel("Servers (M = 8 GPUs/NICs)")
    axes[1].set_ylabel("Bottleneck (records)")
    axes[1].set_xticks(servers)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()
    figure.savefig(output_dir / "01_fast_dlb_scheduler_and_bottleneck.png", dpi=180)
    plt.close(figure)


def plot_staging_and_control(rows: list[dict[str, float]], output_dir: Path) -> None:
    plt = get_pyplot()
    servers = [int(row["servers"]) for row in rows]
    positions = list(range(len(rows)))

    figure, axes = plt.subplots(2, 1, figsize=(10, 8))
    figure.subplots_adjust(left=0.18, right=0.98, bottom=0.10, top=0.95, hspace=0.70)
    width = 0.36
    dlb_staging = [row["dlb_source_staging_records"] for row in rows]
    fast_staging = [row["fast_source_staging_records"] for row in rows]
    dlb_bars = axes[0].bar([position - width / 2 for position in positions], dlb_staging, width,
                           color=DLB_COLOR, label="DLB source staging")
    fast_bars = axes[0].bar([position + width / 2 for position in positions], fast_staging, width,
                            color=FAST_COLOR, label="FAST source staging")
    annotate_bars(axes[0], dlb_bars, dlb_staging)
    annotate_bars(axes[0], fast_bars, fast_staging)
    axes[0].set_title("Source-Side NVLink Staging")
    axes[0].set_ylabel("Records")
    axes[0].set_xticks(positions, servers)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    width = 0.25
    dlb_counters = [row["dlb_local_counters_per_server"] for row in rows]
    fast_matrix = [row["fast_global_matrix_values"] for row in rows]
    fast_stages = [row["fast_birkhoff_stages"] for row in rows]
    dlb_bars = axes[1].bar([position - width for position in positions], dlb_counters, width,
                           color=DLB_COLOR, label="DLB local counters/server")
    matrix_bars = axes[1].bar(positions, fast_matrix, width,
                              color=FAST_COLOR, label="FAST global SxS matrix")
    stage_bars = axes[1].bar([position + width for position in positions], fast_stages, width,
                              color=FAST_STAGE_COLOR, label="FAST Birkhoff stages")
    annotate_bars(axes[1], dlb_bars, dlb_counters)
    annotate_bars(axes[1], matrix_bars, fast_matrix)
    annotate_bars(axes[1], stage_bars, fast_stages)
    axes[1].set_title("Control-Plane Metadata")
    axes[1].set_xlabel("Servers (M = 8)")
    axes[1].set_ylabel("Count")
    axes[1].set_xticks(positions, servers)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()
    figure.savefig(output_dir / "02_fast_dlb_staging_and_control.png", dpi=180)
    plt.close(figure)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: plot_fast_dlb_comparison.py benchmark.csv output_directory")
    csv_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(csv_path)
    if not rows:
        raise ValueError("benchmark CSV contains no rows")
    plot_scheduler_and_bottleneck(rows, output_dir)
    plot_staging_and_control(rows, output_dir)


if __name__ == "__main__":
    main()
