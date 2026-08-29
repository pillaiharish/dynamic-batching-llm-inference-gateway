#!/usr/bin/env python3
"""Generate dependency-free SVG plots from summary.csv; the CSV remains authoritative."""

from __future__ import annotations

import argparse
import csv
import html
import statistics
from collections import defaultdict
from pathlib import Path

PLOTS = {
    "e2e_p95_ms": "E2E p95 vs concurrency",
    "ttft_p95_ms": "TTFT p95 vs concurrency",
    "request_throughput_rps": "Request throughput vs concurrency",
    "output_throughput_tps": "Output token throughput vs concurrency",
    "mean_batch_size": "Mean batch size vs concurrency",
    "batch_wait_p95_ms": "Batch wait p95 vs concurrency",
    "error_rate": "Error rate vs concurrency",
}
COLORS = {"direct": "#2563eb", "gateway_no_batch": "#d97706", "gateway_batch": "#059669"}


def render(path: Path, title: str, series: dict[str, list[tuple[int, float]]]) -> None:
    width, height = 800, 480
    left, top, right, bottom = 80, 55, 30, 70
    points = [point for values in series.values() for point in values]
    if not points:
        return
    x_values = sorted({point[0] for point in points})
    y_max = max(point[1] for point in points) or 1

    def x(value: int) -> float:
        index = x_values.index(value)
        return (
            left
            if len(x_values) == 1
            else left + index * (width - left - right) / (len(x_values) - 1)
        )

    def y(value: float) -> float:
        return height - bottom - value * (height - top - bottom) / y_max

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="sans-serif" '
        f'font-size="18">{html.escape(title)}</text>',
        f'<path d="M {left} {top} V {height - bottom} H {width - right}" '
        'fill="none" stroke="#111827"/>',
    ]
    for value in x_values:
        svg.append(
            f'<text x="{x(value)}" y="{height - bottom + 24}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="12">{value}</text>'
        )
    for step in range(6):
        value = y_max * step / 5
        position = y(value)
        svg.append(f'<path d="M {left} {position} H {width - right}" stroke="#e5e7eb"/>')
        svg.append(
            f'<text x="{left - 8}" y="{position + 4}" text-anchor="end" '
            f'font-family="sans-serif" font-size="11">{value:.2f}</text>'
        )
    for index, (label, values) in enumerate(sorted(series.items())):
        color = COLORS.get(label, "#7c3aed")
        coordinates = " ".join(f"{x(point_x)},{y(point_y)}" for point_x, point_y in values)
        svg.append(
            f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2"/>'
        )
        for point_x, point_y in values:
            svg.append(f'<circle cx="{x(point_x)}" cy="{y(point_y)}" r="3" fill="{color}"/>')
        svg.append(
            f'<text x="{left + index * 180}" y="{height - 18}" '
            f'font-family="sans-serif" font-size="12" fill="{color}">'
            f"{html.escape(label)}</text>"
        )
    svg.append(
        f'<text x="{width / 2}" y="{height - 40}" text-anchor="middle" '
        'font-family="sans-serif" font-size="12">Concurrency</text>'
    )
    svg.append("</svg>\n")
    path.write_text("\n".join(svg), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args()
    rows = list(csv.DictReader(arguments.csv.open(encoding="utf-8")))
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    for mode in sorted({row["mode"] for row in rows}):
        for metric, title in PLOTS.items():
            grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
            for row in rows:
                if row["mode"] == mode and row.get(metric) not in {None, ""}:
                    grouped[(row["target"], int(row["concurrency"]))].append(float(row[metric]))
            series: dict[str, list[tuple[int, float]]] = defaultdict(list)
            for (target, concurrency), values in grouped.items():
                series[target].append((concurrency, statistics.median(values)))
            for values in series.values():
                values.sort()
            render(arguments.output_dir / f"{mode}-{metric}.svg", title, series)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
