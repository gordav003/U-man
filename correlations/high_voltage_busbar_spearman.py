from __future__ import annotations

"""Spearmanova korelacija 15-minutnih sprememb napetosti VN zbiralk.

Zbiralka je v tej analizi definirana kot kombinacija lokacije RTP in
napetostnega nivoja (110, 220 ali 400 kV), za katero obstaja dovolj veljavnih
napetostnih meritev. Vsak fizicni transformator je
uporabljen samo na najvisjem VN-nivoju, ki je zanj prisoten v podatkih. Tako je
na primer 220/110-kV transformator upostevan samo na 220 kV, 400/110-kV
transformator pa samo na 400 kV. Napetost zbiralke v posameznem trenutku je
mediana vseh veljavnih napetosti tako izbranih transformatorjev na isti
lokaciji in istem nivoju. Sprememba je izracunana samo, kadar obstajata meritvi
obeh tocno dolocenih trenutkov:

    dU(t) = U(t + 15 min) - U(t)

Manjkajoce meritve se ne interpolirajo. Casovna os se ob vsaki vrzeli, vecji
od dovoljenega koraka, razdeli na zvezne segmente. Za vsak segment se izdela
locena Spearmanova korelacijska matrika; podatki locenih segmentov se nikoli
ne zdruzijo v isti koeficient.

Privzeti zagon iz korena projekta:

    python -m correlations.high_voltage_busbar_spearman

Potrebne knjiznice: polars, numpy, scipy in matplotlib.
"""

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import polars as pl
from scipy.stats import ConstantInputWarning, spearmanr


HV_LEVELS_KV = (110, 220, 400)
# Znani kanali, katerih vrednosti so stevilcno videti smiselne, vendar meritve
# niso veljavne za korelacijsko analizo.
EXCLUDED_LOCATIONS = {"LENART"}
VALID_U_MIN_PU = 0.5
VALID_U_MAX_PU = 1.5
DELTA_MINUTES = 15
DEFAULT_MIN_PAIR_POINTS = 20
DEFAULT_MAX_GAP_MINUTES = 15
DEFAULT_MIN_SEGMENT_POINTS = 20


@dataclass(frozen=True)
class ContinuousSegment:
    number: int
    start: datetime
    end: datetime
    n_timestamps: int

    @property
    def tag(self) -> str:
        return (
            f"segment_{self.number:03d}_"
            f"{self.start:%Y%m%d_%H%M}_do_{self.end:%Y%m%d_%H%M}"
        )

    @property
    def label(self) -> str:
        return f"{self.start:%Y-%m-%d %H:%M} do {self.end:%Y-%m-%d %H:%M}"


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_input_path() -> Path:
    return (
        project_root()
        / "Uman meritve"
        / "Pridobljeno in urejeno"
        / "urejeno"
        / "Uman_parquet"
        / "transformers_wide.parquet"
    )


def default_output_dir() -> Path:
    return (
        project_root()
        / "Uman meritve"
        / "korelacija_VN_zbiralk_spearman"
        / "zvezni_segmenti"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Izracuna Spearmanovo korelacijsko matriko tocnih 15-minutnih "
            "sprememb napetosti zbiralk na 110, 220 in 400 kV."
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
        help="Mapa za CSV-tabele, PNG/SVG heatmap in povzetek.",
    )
    parser.add_argument(
        "--min-pair-points",
        type=int,
        default=DEFAULT_MIN_PAIR_POINTS,
        help=(
            "Najmanjse stevilo skupnih veljavnih 15-minutnih sprememb za "
            "objavo koeficienta (privzeto: 20)."
        ),
    )
    parser.add_argument(
        "--max-gap-minutes",
        type=int,
        default=DEFAULT_MAX_GAP_MINUTES,
        help=(
            "Najvecji dovoljen razmik med zaporednima casoma istega "
            "zveznega segmenta (privzeto: 15 min)."
        ),
    )
    parser.add_argument(
        "--min-segment-points",
        type=int,
        default=DEFAULT_MIN_SEGMENT_POINTS,
        help=(
            "Najmanjse stevilo casovnih tock, da se zvezni segment analizira "
            "(privzeto: 20)."
        ),
    )
    parser.add_argument(
        "--annotate-max-busbars",
        type=int,
        default=45,
        help=(
            "Ko je zbiralk najvec toliko, heatmap vsebuje tudi stevilke v "
            "celicah. Pri vecjih matrikah ostanejo koeficienti v CSV-tabeli."
        ),
    )
    return parser.parse_args()


def validate_schema(parquet_path: Path) -> None:
    required = {"time", "component_id", "napetost_kv", "lokacija_od", "U"}
    available = set(pl.scan_parquet(parquet_path).collect_schema().names())
    missing = sorted(required - available)
    if missing:
        raise ValueError(
            "V vhodnem Parquetu manjkajo stolpci: " + ", ".join(missing)
        )


def load_busbar_voltages(parquet_path: Path) -> pl.DataFrame:
    """Vrne eno robustno napetost na cas, lokacijo in najvisji VN-nivo."""
    valid_voltage = (
        pl.col("U").is_not_null()
        & pl.col("U").is_finite()
        & (
            pl.col("U")
            / pl.col("napetost_kv").cast(pl.Float64)
        ).is_between(VALID_U_MIN_PU, VALID_U_MAX_PU, closed="both")
    )

    hv_rows = (
        pl.scan_parquet(parquet_path)
        .filter(pl.col("napetost_kv").is_in(HV_LEVELS_KV))
        .filter(pl.col("lokacija_od").is_not_null())
        .filter(pl.col("lokacija_od").str.strip_chars() != "")
        .filter(
            ~pl.col("lokacija_od")
            .str.strip_chars()
            .str.to_uppercase()
            .is_in(EXCLUDED_LOCATIONS)
        )
        .with_columns(
            pl.col("objekt")
            .fill_null(pl.col("component_id"))
            .alias("fizicni_trafo")
        )
        # Isti fizicni trafo je v viru lahko zapisan na obeh straneh, npr.
        # BERICEVO TR211 na 110 in 220 kV. Obdrzimo samo najvisji nivo.
        .with_columns(
            pl.col("napetost_kv")
            .max()
            .over(["lokacija_od", "fizicni_trafo"])
            .alias("najvisji_nivo_kv")
        )
        .filter(pl.col("napetost_kv") == pl.col("najvisji_nivo_kv"))
        .filter(valid_voltage)
    )

    return (
        hv_rows
        .group_by(["time", "lokacija_od", "napetost_kv"])
        .agg(
            pl.col("U").median().alias("U_kV"),
            pl.col("component_id").n_unique().alias("n_transformers_at_time"),
        )
        .with_columns(
            (
                pl.col("lokacija_od")
                + pl.lit(" | ")
                + pl.col("napetost_kv").cast(pl.String)
                + pl.lit(" kV")
            ).alias("zbiralka")
        )
        .sort(["napetost_kv", "lokacija_od", "time"])
        .collect()
    )


def exact_15_minute_changes(busbar_voltages: pl.DataFrame) -> pl.DataFrame:
    current = busbar_voltages.select(
        "time", "zbiralka", pl.col("U_kV").alias("U_t_kV")
    )
    future = busbar_voltages.select(
        (pl.col("time") - pl.duration(minutes=DELTA_MINUTES)).alias("time"),
        "zbiralka",
        pl.col("U_kV").alias("U_t_plus_15_kV"),
    )

    return (
        current.join(future, on=["time", "zbiralka"], how="inner")
        .with_columns(
            (pl.col("U_t_plus_15_kV") - pl.col("U_t_kV")).alias("dU_kV")
        )
        .select("time", "zbiralka", "dU_kV")
        .sort(["time", "zbiralka"])
    )


def find_continuous_segments(
    busbar_voltages: pl.DataFrame,
    max_gap_minutes: int,
    min_segment_points: int,
) -> list[ContinuousSegment]:
    """Razdeli skupno casovno os na strogo locene merilne segmente."""
    times = (
        busbar_voltages.select("time")
        .unique()
        .sort("time")
        .get_column("time")
        .to_list()
    )
    if not times:
        return []

    maximum_gap = timedelta(minutes=max_gap_minutes)
    raw_segments: list[list[datetime]] = [[times[0]]]
    for previous, current in zip(times, times[1:]):
        if current - previous > maximum_gap:
            raw_segments.append([])
        raw_segments[-1].append(current)

    result = []
    for segment_times in raw_segments:
        if len(segment_times) < min_segment_points:
            continue
        result.append(
            ContinuousSegment(
                number=len(result) + 1,
                start=segment_times[0],
                end=segment_times[-1],
                n_timestamps=len(segment_times),
            )
        )
    return result


def rows_in_segment(
    busbar_voltages: pl.DataFrame, segment: ContinuousSegment
) -> pl.DataFrame:
    return busbar_voltages.filter(
        pl.col("time").is_between(segment.start, segment.end, closed="both")
    )


def write_segments_index(path: Path, segments: list[ContinuousSegment]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(
            ["segment", "zacetek", "konec", "stevilo_casovnih_tock"]
        )
        for segment in segments:
            writer.writerow(
                [
                    segment.tag,
                    segment.start.isoformat(sep=" "),
                    segment.end.isoformat(sep=" "),
                    segment.n_timestamps,
                ]
            )


def ordered_busbars(busbar_voltages: pl.DataFrame) -> list[str]:
    return (
        busbar_voltages.select("napetost_kv", "lokacija_od", "zbiralka")
        .unique()
        .sort(["napetost_kv", "lokacija_od"])
        .get_column("zbiralka")
        .to_list()
    )


def eligible_busbars_for_segment(
    segment_voltages: pl.DataFrame,
    changes: pl.DataFrame,
    min_pair_points: int,
) -> tuple[list[str], pl.DataFrame, pl.DataFrame]:
    """Obdrzi samo zbiralke z dovolj veljavnimi 15-minutnimi spremembami U."""
    eligible = (
        changes.group_by("zbiralka")
        .agg(pl.col("dU_kV").is_finite().sum().alias("n_veljavnih_dU"))
        .filter(pl.col("n_veljavnih_dU") >= min_pair_points)
        .select("zbiralka")
    )
    filtered_voltages = segment_voltages.join(
        eligible, on="zbiralka", how="inner"
    )
    filtered_changes = changes.join(eligible, on="zbiralka", how="inner")
    return (
        ordered_busbars(filtered_voltages),
        filtered_voltages,
        filtered_changes,
    )


def wide_change_matrix(
    changes: pl.DataFrame, busbars: list[str]
) -> tuple[list, np.ndarray]:
    wide = changes.pivot(
        index="time",
        on="zbiralka",
        values="dU_kV",
        aggregate_function="first",
    ).sort("time")

    for busbar in busbars:
        if busbar not in wide.columns:
            wide = wide.with_columns(pl.lit(None, dtype=pl.Float64).alias(busbar))

    wide = wide.select(["time", *busbars])
    times = wide.get_column("time").to_list()
    values = np.column_stack(
        [wide.get_column(name).to_numpy() for name in busbars]
    ).astype(float, copy=False)
    return times, values


def spearman_matrix(
    values: np.ndarray, min_pair_points: int
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(values)
    pair_counts = valid.astype(np.int32).T @ valid.astype(np.int32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        result = spearmanr(values, axis=0, nan_policy="omit")

    correlation = np.asarray(result.statistic, dtype=float)
    if correlation.ndim == 0:
        correlation = correlation.reshape(1, 1)
    correlation[pair_counts < min_pair_points] = np.nan
    return correlation, pair_counts


def write_square_csv(
    path: Path,
    busbars: list[str],
    matrix: np.ndarray,
    integer: bool = False,
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["zbiralka", *busbars])
        for name, row in zip(busbars, matrix):
            if integer:
                values = [str(int(value)) for value in row]
            else:
                values = ["" if not np.isfinite(value) else f"{value:.6f}" for value in row]
            writer.writerow([name, *values])


def build_metadata(
    busbar_voltages: pl.DataFrame, changes: pl.DataFrame
) -> pl.DataFrame:
    voltage_summary = (
        busbar_voltages.group_by(
            ["zbiralka", "napetost_kv", "lokacija_od"]
        )
        .agg(
            pl.len().alias("n_casovnih_tock_U"),
            pl.col("n_transformers_at_time").max().alias(
                "max_transformatorjev_v_casu"
            ),
            pl.col("time").min().alias("prva_meritev"),
            pl.col("time").max().alias("zadnja_meritev"),
        )
    )
    change_summary = changes.group_by("zbiralka").agg(
        pl.len().alias("n_tocnih_15min_razlik"),
        pl.col("dU_kV").n_unique().alias("n_razlicnih_dU"),
    )
    return (
        voltage_summary.join(change_summary, on="zbiralka", how="left")
        .with_columns(
            pl.col("n_tocnih_15min_razlik").fill_null(0),
            pl.col("n_razlicnih_dU").fill_null(0),
        )
        .sort(["napetost_kv", "lokacija_od"])
    )


def draw_heatmap(
    correlation: np.ndarray,
    busbars: list[str],
    png_path: Path,
    svg_path: Path,
    annotate_max_busbars: int,
    period_label: str,
) -> None:
    n = len(busbars)
    figure_size = max(18.0, min(54.0, 0.31 * n))
    font_size = max(3.2, min(7.0, 520.0 / max(n, 1)))
    cmap = LinearSegmentedColormap.from_list(
        "blue_white_orange", ["#2463A8", "#F7F8FA", "#D97706"]
    )
    cmap.set_bad("#D9DDE3")

    fig, ax = plt.subplots(figsize=(figure_size, figure_size), dpi=180)
    image = ax.imshow(
        np.ma.masked_invalid(correlation),
        cmap=cmap,
        vmin=-1,
        vmax=1,
        interpolation="nearest",
        aspect="equal",
    )
    ax.set_title(
        "Spearman correlation of 15-minute HV busbar voltage changes",
        fontsize=max(12, font_size * 2.2),
        color="#20252B",
        pad=18,
    )
    ax.text(
        0,
        1.012,
        "ΔU(t) = U(t + 15 min) - U(t); HV levels 110, 220 and 400 kV; "
        f"continuous segment: {period_label}",
        transform=ax.transAxes,
        fontsize=max(8, font_size * 1.45),
        color="#5B6470",
        va="bottom",
    )
    positions = np.arange(n)
    ax.set_xticks(positions)
    ax.set_yticks(positions)
    ax.set_xticklabels(busbars, rotation=90, fontsize=font_size)
    ax.set_yticklabels(busbars, fontsize=font_size)
    ax.tick_params(axis="both", length=0, pad=2)

    boundaries = []
    previous_level = busbars[0].split(" | ")[-1] if busbars else None
    for index, name in enumerate(busbars[1:], start=1):
        level = name.split(" | ")[-1]
        if level != previous_level:
            boundaries.append(index - 0.5)
            previous_level = level
    for boundary in boundaries:
        ax.axhline(boundary, color="#20252B", linewidth=0.9)
        ax.axvline(boundary, color="#20252B", linewidth=0.9)

    if n <= annotate_max_busbars:
        for row in range(n):
            for column in range(n):
                value = correlation[row, column]
                if np.isfinite(value):
                    text_color = "white" if abs(value) >= 0.62 else "#20252B"
                    ax.text(
                        column,
                        row,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=max(4.5, font_size * 0.82),
                        color=text_color,
                    )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.012)
    colorbar.set_label("Spearman coefficient", color="#20252B")
    colorbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xlabel("ΔU / kV")
    ax.set_ylabel("ΔU / kV")
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)


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
    if args.annotate_max_busbars < 0:
        raise ValueError("--annotate-max-busbars ne sme biti negativen.")

    validate_schema(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    busbar_voltages = load_busbar_voltages(input_path)
    busbars = ordered_busbars(busbar_voltages)
    if not busbars:
        raise ValueError("V vhodnih podatkih ni RTP z veljavnimi VN napetostmi.")
    segments = find_continuous_segments(
        busbar_voltages,
        max_gap_minutes=args.max_gap_minutes,
        min_segment_points=args.min_segment_points,
    )
    if not segments:
        raise ValueError("Ni dovolj dolgega zveznega segmenta za analizo.")

    write_segments_index(output_dir / "segmenti.csv", segments)
    root_summary = [
        "SPEARMANOVA KORELACIJA VN ZBIRALK PO ZVEZNIH SEGMENTIH",
        f"Vhod: {input_path}",
        f"VN nivoji: {', '.join(map(str, HV_LEVELS_KV))} kV",
        f"Definicija: dU(t) = U(t + {DELTA_MINUTES} min) - U(t)",
        f"Prekinitev segmenta pri vrzeli > {args.max_gap_minutes} min",
        f"Minimalno casovnih tock segmenta: {args.min_segment_points}",
        f"Stevilo analiziranih segmentov: {len(segments)}",
        "Vsak segment ima lastno matriko; segmenti se ne zdruzujejo.",
    ]
    (output_dir / "izracun_povzetek.txt").write_text(
        "\n".join(root_summary) + "\n", encoding="utf-8"
    )

    for segment in segments:
        segment_dir = output_dir / segment.tag
        segment_dir.mkdir(parents=True, exist_ok=True)
        segment_voltages = rows_in_segment(busbar_voltages, segment)
        changes = exact_15_minute_changes(segment_voltages)
        segment_busbars, segment_voltages, changes = eligible_busbars_for_segment(
            segment_voltages, changes, args.min_pair_points
        )
        if not segment_busbars:
            (segment_dir / "izracun_povzetek.txt").write_text(
                "V segmentu ni RTP z dovolj veljavnimi napetostnimi meritvami.\n",
                encoding="utf-8",
            )
            print(
                f"Segment {segment.number}/{len(segments)} preskocen: "
                "ni RTP z dovolj veljavnimi napetostmi."
            )
            continue

        times, values = wide_change_matrix(changes, segment_busbars)
        correlation, pair_counts = spearman_matrix(values, args.min_pair_points)
        metadata = build_metadata(segment_voltages, changes)

        correlation_path = segment_dir / "spearman_korelacijska_matrika.csv"
        counts_path = segment_dir / "stevilo_skupnih_15min_parov.csv"
        metadata_path = segment_dir / "zbiralke_metadata.csv"
        png_path = segment_dir / "spearman_heatmap_dU_15min.png"
        svg_path = segment_dir / "spearman_heatmap_dU_15min.svg"

        write_square_csv(correlation_path, segment_busbars, correlation)
        write_square_csv(counts_path, segment_busbars, pair_counts, integer=True)
        metadata.write_csv(metadata_path, separator=";", include_bom=True)
        draw_heatmap(
            correlation,
            segment_busbars,
            png_path,
            svg_path,
            args.annotate_max_busbars,
            segment.label,
        )

        finite_off_diagonal = correlation[
            ~np.eye(len(segment_busbars), dtype=bool) & np.isfinite(correlation)
        ]
        summary_lines = [
            "SPEARMANOVA KORELACIJA VN ZBIRALK - ZVEZNI SEGMENT",
            f"Segment: {segment.label}",
            f"Casovnih tock segmenta: {segment.n_timestamps}",
            f"Casov z vsaj eno tocno 15-minutno razliko: {len(times)}",
            f"Minimalno skupnih razlik na par: {args.min_pair_points}",
            f"Stevilo RTP/zbiralk z veljavnimi napetostmi: {len(segment_busbars)}",
            f"Veljavnih izven-diagonalnih koeficientov: "
            f"{finite_off_diagonal.size}",
            "Agregacija zbiralke: mediana veljavnih napetosti transformatorjev",
            "Trafo z vec VN zapisi: samo na najvisjem nazivnem nivoju",
            "Manjkajoce vrednosti: brez interpolacije; parno izlocanje",
        ]
        (segment_dir / "izracun_povzetek.txt").write_text(
            "\n".join(summary_lines) + "\n", encoding="utf-8"
        )

        print(
            f"Segment {segment.number}/{len(segments)} koncan: "
            f"{segment.label}"
        )

    print("Izracun vseh zveznih segmentov je koncan.")
    print(f"Izhodna mapa: {output_dir}")


if __name__ == "__main__":
    main()
