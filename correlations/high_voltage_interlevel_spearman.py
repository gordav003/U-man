from __future__ import annotations

"""Loceni heatmapi Spearmanove korelacije 110-220 in 110-400 kV.

Skripta uporablja enako pripravo podatkov kot
high_voltage_busbar_spearman.py:

* vsak fizicni transformator je upostevan samo na najvisjem VN-nivoju;
* napetost zbiralke je mediana veljavnih napetosti vseh izbranih trafov;
* dU(t) = U(t + 15 min) - U(t) samo za tocno obstojeca casa;
* casovna os se razdeli na zvezne segmente in vsak segment dobi locena izhoda;
* korelacija je izkljucno Spearmanova in uporablja parno veljavne podatke.

Privzeti zagon iz korena projekta:

    python -m correlations.high_voltage_interlevel_spearman
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

try:
    from .high_voltage_busbar_spearman import (
        DEFAULT_MAX_GAP_MINUTES,
        DEFAULT_MIN_PAIR_POINTS,
        DEFAULT_MIN_SEGMENT_POINTS,
        default_input_path,
        exact_15_minute_changes,
        find_continuous_segments,
        load_busbar_voltages,
        ordered_busbars,
        project_root,
        rows_in_segment,
        spearman_matrix,
        validate_schema,
        wide_change_matrix,
        write_segments_index,
    )
except ImportError:  # Support direct execution from this directory.
    from high_voltage_busbar_spearman import (
        DEFAULT_MAX_GAP_MINUTES,
        DEFAULT_MIN_PAIR_POINTS,
        DEFAULT_MIN_SEGMENT_POINTS,
        default_input_path,
        exact_15_minute_changes,
        find_continuous_segments,
        load_busbar_voltages,
        ordered_busbars,
        project_root,
        rows_in_segment,
        spearman_matrix,
        validate_schema,
        wide_change_matrix,
        write_segments_index,
    )


def default_output_dir() -> Path:
    return (
        project_root()
        / "Uman meritve"
        / "korelacija_VN_med_nivoji_spearman"
        / "zvezni_segmenti"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ustvari locena, anotirana heatmapa Spearmanove korelacije "
            "15-minutnih dU za 110-220 in 110-400 kV."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input_path(),
        help="Vhodni transformers_wide.parquet.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir(),
        help="Izhodna mapa za CSV, PNG in SVG datoteke.",
    )
    parser.add_argument(
        "--min-pair-points",
        type=int,
        default=DEFAULT_MIN_PAIR_POINTS,
        help="Najmanj skupnih 15-minutnih razlik za objavo koeficienta.",
    )
    parser.add_argument(
        "--max-gap-minutes",
        type=int,
        default=DEFAULT_MAX_GAP_MINUTES,
        help="Prekinitev zveznega segmenta pri vecji casovni vrzeli.",
    )
    parser.add_argument(
        "--min-segment-points",
        type=int,
        default=DEFAULT_MIN_SEGMENT_POINTS,
        help="Najmanj casovnih tock analiziranega zveznega segmenta.",
    )
    return parser.parse_args()


def busbars_at_level(busbars: list[str], level_kv: int) -> list[str]:
    suffix = f" | {level_kv} kV"
    return [name for name in busbars if name.endswith(suffix)]


def short_label(busbar: str) -> str:
    return busbar.split(" | ", maxsplit=1)[0].replace("_", " ")


def extract_cross_level_matrix(
    matrix: np.ndarray,
    all_busbars: list[str],
    row_level_kv: int,
    column_level_kv: int,
) -> tuple[list[str], list[str], np.ndarray]:
    rows = busbars_at_level(all_busbars, row_level_kv)
    columns = busbars_at_level(all_busbars, column_level_kv)
    index = {name: position for position, name in enumerate(all_busbars)}
    row_indices = [index[name] for name in rows]
    column_indices = [index[name] for name in columns]
    return rows, columns, matrix[np.ix_(row_indices, column_indices)]


def write_rectangular_csv(
    path: Path,
    row_busbars: list[str],
    column_busbars: list[str],
    matrix: np.ndarray,
    integer: bool = False,
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["zbiralka", *column_busbars])
        for name, values in zip(row_busbars, matrix):
            if integer:
                formatted = [str(int(value)) for value in values]
            else:
                formatted = [
                    "" if not np.isfinite(value) else f"{value:.6f}"
                    for value in values
                ]
            writer.writerow([name, *formatted])


def draw_readable_heatmap(
    correlation: np.ndarray,
    pair_counts: np.ndarray,
    row_busbars: list[str],
    column_busbars: list[str],
    row_level_kv: int,
    column_level_kv: int,
    min_pair_points: int,
    period_label: str,
    png_path: Path,
    svg_path: Path,
) -> None:
    n_rows, n_columns = correlation.shape
    figure_width = max(10.5, 2.0 + 1.25 * n_columns)
    figure_height = max(12.0, 3.2 + 0.29 * n_rows)
    cmap = LinearSegmentedColormap.from_list(
        "blue_white_orange", ["#2463A8", "#F7F8FA", "#D97706"]
    )
    cmap.set_bad("#D9DDE3")

    fig, ax = plt.subplots(
        figsize=(figure_width, figure_height),
        dpi=180,
        facecolor="#FFFFFF",
    )
    image = ax.imshow(
        np.ma.masked_invalid(correlation),
        cmap=cmap,
        vmin=-1,
        vmax=1,
        interpolation="nearest",
        aspect="auto",
    )

    ax.set_title(
        f"Spearman correlation of ΔU: {row_level_kv} kV versus "
        f"{column_level_kv} kV",
        fontsize=15,
        color="#20252B",
        loc="left",
        pad=34,
        fontweight="bold",
    )
    ax.text(
        0,
        1.012,
        "ΔU(t) = U(t + 15 min) - U(t) | cell values are "
        f"Spearman coefficients | segment: {period_label}",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#5B6470",
        va="bottom",
    )
    ax.text(
        0,
        1.001,
        f"Sivo = manj kot {min_pair_points} skupnih veljavnih razlik",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#747D88",
        va="top",
    )

    ax.set_xticks(np.arange(n_columns))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(
        [f"{short_label(name)}\n{column_level_kv} kV" for name in column_busbars],
        fontsize=9,
        fontweight="bold",
    )
    ax.set_yticklabels(
        [short_label(name) for name in row_busbars],
        fontsize=6.2,
    )
    ax.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.tick_params(axis="x", length=0, pad=7)
    ax.set_ylabel("ΔU / kV", fontsize=10, labelpad=10)
    ax.set_xlabel("ΔU / kV", fontsize=10, labelpad=10)
    ax.xaxis.set_label_position("top")

    ax.set_xticks(np.arange(-0.5, n_columns, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="#FFFFFF", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_color("#A7AFB8")
        spine.set_linewidth(0.8)

    for row in range(n_rows):
        for column in range(n_columns):
            value = correlation[row, column]
            if not np.isfinite(value):
                label = "–"
                color = "#5B6470"
            else:
                label = f"{value:.2f}"
                color = "#FFFFFF" if abs(value) >= 0.72 else "#20252B"
            ax.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                fontsize=6.1 if n_columns <= 4 else 5.7,
                color=color,
                fontfamily="DejaVu Sans Mono",
            )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.035)
    colorbar.set_label("Spearman coefficient", fontsize=9)
    colorbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    colorbar.ax.tick_params(labelsize=8)
    colorbar.outline.set_edgecolor("#A7AFB8")

    fig.subplots_adjust(left=0.29, right=0.91, top=0.91, bottom=0.035)
    fig.savefig(png_path, bbox_inches="tight", facecolor="#FFFFFF")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close(fig)


def create_comparison_outputs(
    correlation: np.ndarray,
    pair_counts: np.ndarray,
    all_busbars: list[str],
    row_level_kv: int,
    column_level_kv: int,
    output_dir: Path,
    min_pair_points: int,
    period_label: str,
) -> tuple[Path, Path, Path, Path]:
    row_busbars, column_busbars, cross_correlation = extract_cross_level_matrix(
        correlation,
        all_busbars,
        row_level_kv,
        column_level_kv,
    )
    _, _, cross_counts = extract_cross_level_matrix(
        pair_counts,
        all_busbars,
        row_level_kv,
        column_level_kv,
    )
    stem = f"spearman_{row_level_kv}_{column_level_kv}_dU_15min"
    csv_path = output_dir / f"{stem}.csv"
    counts_path = output_dir / f"{stem}_stevilo_parov.csv"
    png_path = output_dir / f"{stem}_heatmap.png"
    svg_path = output_dir / f"{stem}_heatmap.svg"

    write_rectangular_csv(csv_path, row_busbars, column_busbars, cross_correlation)
    write_rectangular_csv(
        counts_path,
        row_busbars,
        column_busbars,
        cross_counts,
        integer=True,
    )
    draw_readable_heatmap(
        cross_correlation,
        cross_counts,
        row_busbars,
        column_busbars,
        row_level_kv,
        column_level_kv,
        min_pair_points,
        period_label,
        png_path,
        svg_path,
    )
    return csv_path, counts_path, png_path, svg_path


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Vhodna datoteka ne obstaja: {input_path}")
    if args.min_pair_points < 2:
        raise ValueError("--min-pair-points mora biti vsaj 2.")
    if args.max_gap_minutes < 1:
        raise ValueError("--max-gap-minutes mora biti vsaj 1.")
    if args.min_segment_points < 2:
        raise ValueError("--min-segment-points mora biti vsaj 2.")

    validate_schema(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    busbar_voltages = load_busbar_voltages(input_path)
    all_busbars = ordered_busbars(busbar_voltages)
    segments = find_continuous_segments(
        busbar_voltages,
        max_gap_minutes=args.max_gap_minutes,
        min_segment_points=args.min_segment_points,
    )
    if not segments:
        raise ValueError("Ni dovolj dolgega zveznega segmenta za analizo.")

    write_segments_index(output_dir / "segmenti.csv", segments)
    summary_lines = [
        "LOCENI SPEARMANOVI HEATMAPA PO ZVEZNIH SEGMENTIH",
        f"Vhod: {input_path}",
        f"Prekinitev segmenta pri vrzeli > {args.max_gap_minutes} min",
        f"Minimalno casovnih tock segmenta: {args.min_segment_points}",
        f"Stevilo analiziranih segmentov: {len(segments)}",
        "Za vsak segment: 110-220 kV in 110-400 kV.",
        "Segmenti se ne zdruzujejo v skupen Spearmanov koeficient.",
    ]
    (output_dir / "izracun_povzetek.txt").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )

    for segment in segments:
        segment_dir = output_dir / segment.tag
        segment_dir.mkdir(parents=True, exist_ok=True)
        segment_voltages = rows_in_segment(busbar_voltages, segment)
        changes = exact_15_minute_changes(segment_voltages)
        _, values = wide_change_matrix(changes, all_busbars)
        correlation, pair_counts = spearman_matrix(
            values, args.min_pair_points
        )

        create_comparison_outputs(
            correlation,
            pair_counts,
            all_busbars,
            110,
            220,
            segment_dir,
            args.min_pair_points,
            segment.label,
        )
        create_comparison_outputs(
            correlation,
            pair_counts,
            all_busbars,
            110,
            400,
            segment_dir,
            args.min_pair_points,
            segment.label,
        )
        print(
            f"Segment {segment.number}/{len(segments)} koncan: "
            f"{segment.label}"
        )

    print("Locena heatmapa vseh zveznih segmentov so izdelana.")
    print(f"Izhodna mapa: {output_dir}")


if __name__ == "__main__":
    main()
