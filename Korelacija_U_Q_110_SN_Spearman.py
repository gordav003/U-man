from __future__ import annotations

"""Spearmanova korelacija 15-minutnih sprememb U in Q zbiralk 110/SN.

Zbiralka je lokacija RTP z vsaj enim strogim transformatorjem 110/SN in z
veljavnimi meritvami U ter Q na 110-kV strani. Strogi
transformator 110/SN ima za isti objekt meritev na 110 kV, vsaj eno meritev na
nivoju pod 110 kV in nobene meritve na nivoju nad 110 kV. Tako so izloceni
transformatorji 220/110 kV, 400/110 kV in njihove terciarne strani.

Za vsak cas in lokacijo velja:

* U_110 je mediana veljavnih 110-kV napetosti transformatorjev 110/SN;
* Q je vsota veljavnih Q na 110-kV straneh vseh transformatorjev 110/SN;
* dU(t) = U(t + 15 min) - U(t);
* dQ(t) = Q(t + 15 min) - Q(t).

Skripta za vsak zvezni casovni segment izdela celotno matriko Spearmanovih
korelacij dU vseh zbiralk proti dQ vseh lokacij. Posebej shrani tudi diagonalo
matrike, torej korelacijo dU in dQ na isti lokaciji. Manjkajoce meritve se ne
interpolirajo, meritve iz locenih segmentov pa se nikoli ne zdruzijo.

Privzeti zagon iz korena projekta:

    python U-man/Korelacija_U_Q_110_SN_Spearman.py

Potrebne knjiznice: polars, numpy, scipy in matplotlib.
"""

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import polars as pl
from scipy.stats import ConstantInputWarning, spearmanr


HV_VOLTAGE_KV = 110
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
        / "korelacija_U_Q_110_SN_spearman"
        / "zvezni_segmenti"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Izracuna Spearmanovo korelacijo tocnih 15-minutnih sprememb "
            "110-kV napetosti in vsote Q za vse stroge zbiralke 110/SN."
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
        help="Mapa za CSV-tabele, PNG/SVG heatmap in povzetke.",
    )
    parser.add_argument(
        "--min-pair-points",
        type=int,
        default=DEFAULT_MIN_PAIR_POINTS,
        help="Najmanj skupnih veljavnih parov dU-dQ (privzeto: 20).",
    )
    parser.add_argument(
        "--max-gap-minutes",
        type=int,
        default=DEFAULT_MAX_GAP_MINUTES,
        help="Najvecji razmik znotraj zveznega segmenta (privzeto: 15 min).",
    )
    parser.add_argument(
        "--min-segment-points",
        type=int,
        default=DEFAULT_MIN_SEGMENT_POINTS,
        help="Najmanj casovnih tock analiziranega segmenta (privzeto: 20).",
    )
    parser.add_argument(
        "--annotate-max-busbars",
        type=int,
        default=45,
        help="Najvecje stevilo zbiralk za izpis koeficientov v heatmap celicah.",
    )
    return parser.parse_args()


def validate_schema(parquet_path: Path) -> None:
    required = {
        "time",
        "component_id",
        "napetost_kv",
        "lokacija_od",
        "objekt",
        "Q",
        "U",
    }
    available = set(pl.scan_parquet(parquet_path).collect_schema().names())
    missing = sorted(required - available)
    if missing:
        raise ValueError("V vhodnem Parquetu manjkajo stolpci: " + ", ".join(missing))


def load_110_sn_busbars(
    parquet_path: Path,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Vrne casovne vrste zbiralk in stalne podatke o njihovih trafih 110/SN."""
    source = (
        pl.scan_parquet(parquet_path)
        .filter(pl.col("lokacija_od").is_not_null())
        .filter(pl.col("lokacija_od").str.strip_chars() != "")
        .with_columns(
            pl.col("objekt")
            .fill_null(pl.col("component_id"))
            .alias("fizicni_trafo")
        )
    )

    levels = source.select(
        "lokacija_od", "fizicni_trafo", "napetost_kv"
    ).unique()
    strict_keys = (
        levels.group_by(["lokacija_od", "fizicni_trafo"])
        .agg(
            pl.col("napetost_kv").max().alias("najvisji_nivo_kv"),
            (pl.col("napetost_kv") < HV_VOLTAGE_KV)
            .any()
            .alias("ima_SN_stran"),
            pl.col("napetost_kv").sort().unique().alias("nivoji_kv"),
        )
        .filter(pl.col("najvisji_nivo_kv") == HV_VOLTAGE_KV)
        .filter(pl.col("ima_SN_stran"))
    )

    strict_110 = (
        source.filter(pl.col("napetost_kv") == HV_VOLTAGE_KV)
        .join(
            strict_keys.select(
                "lokacija_od", "fizicni_trafo", "nivoji_kv"
            ),
            on=["lokacija_od", "fizicni_trafo"],
            how="inner",
        )
    )

    valid_u = (
        pl.col("U").is_not_null()
        & pl.col("U").is_finite()
        & (pl.col("U") / pl.lit(float(HV_VOLTAGE_KV))).is_between(
            VALID_U_MIN_PU, VALID_U_MAX_PU, closed="both"
        )
    )
    valid_q = pl.col("Q").is_not_null() & pl.col("Q").is_finite()

    # Najprej naredimo eno meritev na fizicni trafo. S tem se isti trafo ne
    # more podvojiti, tudi ce bi imel v viru vec 110-kV zapisov.
    transformer_values = (
        strict_110.group_by(["time", "lokacija_od", "fizicni_trafo"])
        .agg(
            pl.col("U").filter(valid_u).median().alias("U_trafo_kV"),
            pl.col("Q").filter(valid_q).median().alias("Q_trafo_MVAr"),
            pl.col("component_id").n_unique().alias("n_110_zapisov"),
        )
    )

    # Izlocimo RTP, ki imajo v podatkih samo identifikacijo transformatorja,
    # nimajo pa nobene uporabne 110-kV meritve U ali Q. Dodatni filter po
    # segmentih spodaj nato zahteva se dovolj tocnih 15-minutnih sprememb.
    eligible_locations = (
        transformer_values.group_by("lokacija_od")
        .agg(
            pl.col("U_trafo_kV").is_not_null().sum().alias("n_veljavnih_U"),
            pl.col("Q_trafo_MVAr").is_not_null().sum().alias("n_veljavnih_Q"),
        )
        .filter(pl.col("n_veljavnih_U") > 0)
        .filter(pl.col("n_veljavnih_Q") > 0)
        .select("lokacija_od")
    )

    busbars = (
        transformer_values.join(eligible_locations, on="lokacija_od", how="inner")
        .group_by(["time", "lokacija_od"])
        .agg(
            pl.col("U_trafo_kV").median().alias("U_110_kV"),
            pl.col("Q_trafo_MVAr").sum().alias("Q_110_SN_MVAr"),
            pl.col("U_trafo_kV").is_not_null().sum().alias("n_trafov_U"),
            pl.col("Q_trafo_MVAr").is_not_null().sum().alias("n_trafov_Q"),
        )
        .with_columns(
            pl.when(pl.col("n_trafov_Q") > 0)
            .then(pl.col("Q_110_SN_MVAr"))
            .otherwise(None)
            .alias("Q_110_SN_MVAr"),
            (
                pl.col("lokacija_od")
                + pl.lit(" | ")
                + pl.lit(str(HV_VOLTAGE_KV))
                + pl.lit(" kV")
            ).alias("zbiralka"),
        )
        .sort(["lokacija_od", "time"])
        .collect()
    )

    strict_keys_for_metadata = strict_keys.with_columns(
        pl.col("nivoji_kv")
        .list.eval(pl.element().cast(pl.String))
        .list.join(",")
        .alias("nivoji_trafo_kV")
    )
    group_metadata = (
        strict_keys_for_metadata.join(
            eligible_locations, on="lokacija_od", how="inner"
        )
        .group_by("lokacija_od")
        .agg(
            pl.col("fizicni_trafo").n_unique().alias("n_trafov_110_SN"),
            pl.col("fizicni_trafo").sort().str.join(";").alias("transformatorji"),
            pl.col("nivoji_trafo_kV")
            .sort()
            .str.join(";")
            .alias("nivoji_transformatorjev_kV"),
        )
        .with_columns(
            (
                pl.col("lokacija_od")
                + pl.lit(" | ")
                + pl.lit(str(HV_VOLTAGE_KV))
                + pl.lit(" kV")
            ).alias("zbiralka")
        )
        .sort("lokacija_od")
        .collect()
    )
    return busbars, group_metadata


def exact_15_minute_changes(busbars: pl.DataFrame) -> pl.DataFrame:
    current = busbars.select(
        "time",
        "zbiralka",
        pl.col("U_110_kV").alias("U_t_kV"),
        pl.col("Q_110_SN_MVAr").alias("Q_t_MVAr"),
    )
    future = busbars.select(
        (pl.col("time") - pl.duration(minutes=DELTA_MINUTES)).alias("time"),
        "zbiralka",
        pl.col("U_110_kV").alias("U_t_plus_15_kV"),
        pl.col("Q_110_SN_MVAr").alias("Q_t_plus_15_MVAr"),
    )
    return (
        current.join(future, on=["time", "zbiralka"], how="inner")
        .with_columns(
            (pl.col("U_t_plus_15_kV") - pl.col("U_t_kV")).alias("dU_kV"),
            (pl.col("Q_t_plus_15_MVAr") - pl.col("Q_t_MVAr")).alias(
                "dQ_MVAr"
            ),
        )
        .select("time", "zbiralka", "dU_kV", "dQ_MVAr")
        .sort(["time", "zbiralka"])
    )


def find_continuous_segments(
    busbars: pl.DataFrame,
    max_gap_minutes: int,
    min_segment_points: int,
) -> list[ContinuousSegment]:
    times = busbars.select("time").unique().sort("time").get_column("time").to_list()
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


def rows_in_segment(frame: pl.DataFrame, segment: ContinuousSegment) -> pl.DataFrame:
    return frame.filter(
        pl.col("time").is_between(segment.start, segment.end, closed="both")
    )


def ordered_busbars(group_metadata: pl.DataFrame) -> list[str]:
    return group_metadata.sort("lokacija_od").get_column("zbiralka").to_list()


def eligible_busbars_for_segment(
    changes: pl.DataFrame,
    group_metadata: pl.DataFrame,
    min_pair_points: int,
) -> tuple[list[str], pl.DataFrame]:
    """Vrne zbiralke z dovolj veljavnimi dU in dQ v danem segmentu."""
    availability = (
        changes.group_by("zbiralka")
        .agg(
            pl.col("dU_kV").is_finite().sum().alias("n_veljavnih_dU"),
            pl.col("dQ_MVAr").is_finite().sum().alias("n_veljavnih_dQ"),
        )
        .filter(pl.col("n_veljavnih_dU") >= min_pair_points)
        .filter(pl.col("n_veljavnih_dQ") >= min_pair_points)
        .select("zbiralka")
    )
    eligible_metadata = group_metadata.join(
        availability, on="zbiralka", how="inner"
    ).sort("lokacija_od")
    return ordered_busbars(eligible_metadata), eligible_metadata


def wide_values(
    changes: pl.DataFrame, busbars: list[str], value_column: str
) -> np.ndarray:
    wide = changes.pivot(
        index="time",
        on="zbiralka",
        values=value_column,
        aggregate_function="first",
    ).sort("time")
    for busbar in busbars:
        if busbar not in wide.columns:
            wide = wide.with_columns(pl.lit(None, dtype=pl.Float64).alias(busbar))
    wide = wide.select(busbars)
    return np.column_stack(
        [wide.get_column(name).to_numpy() for name in busbars]
    ).astype(float, copy=False)


def spearman_cross_matrix(
    du_values: np.ndarray,
    dq_values: np.ndarray,
    min_pair_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    valid_u = np.isfinite(du_values)
    valid_q = np.isfinite(dq_values)
    pair_counts = valid_u.astype(np.int32).T @ valid_q.astype(np.int32)
    n_busbars = du_values.shape[1]
    combined = np.column_stack([du_values, dq_values])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        result = spearmanr(combined, axis=0, nan_policy="omit")
    full_correlation = np.asarray(result.statistic, dtype=float)
    if n_busbars == 1:
        correlation = full_correlation.reshape(1, 1)
    else:
        correlation = full_correlation[:n_busbars, n_busbars:]
    correlation[pair_counts < min_pair_points] = np.nan
    return correlation, pair_counts


def write_matrix_csv(
    path: Path,
    busbars: list[str],
    matrix: np.ndarray,
    integer: bool = False,
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["dU_zbiralka", *[f"dQ: {name}" for name in busbars]])
        for name, row in zip(busbars, matrix):
            if integer:
                values = [str(int(value)) for value in row]
            else:
                values = [
                    "" if not np.isfinite(value) else f"{value:.6f}"
                    for value in row
                ]
            writer.writerow([f"dU: {name}", *values])


def build_segment_metadata(
    segment_busbars: pl.DataFrame,
    changes: pl.DataFrame,
    group_metadata: pl.DataFrame,
) -> pl.DataFrame:
    measurement_summary = segment_busbars.group_by("zbiralka").agg(
        pl.col("U_110_kV").is_not_null().sum().alias("n_tock_U"),
        pl.col("Q_110_SN_MVAr").is_not_null().sum().alias("n_tock_Q"),
        pl.col("n_trafov_U").max().alias("max_trafov_U_v_casu"),
        pl.col("n_trafov_Q").max().alias("max_trafov_Q_v_casu"),
        pl.col("time").min().alias("prva_meritev"),
        pl.col("time").max().alias("zadnja_meritev"),
    )
    change_summary = changes.group_by("zbiralka").agg(
        pl.col("dU_kV").is_not_null().sum().alias("n_tocnih_dU_15min"),
        pl.col("dQ_MVAr").is_not_null().sum().alias("n_tocnih_dQ_15min"),
        pl.col("dU_kV").n_unique().alias("n_razlicnih_dU"),
        pl.col("dQ_MVAr").n_unique().alias("n_razlicnih_dQ"),
    )
    return (
        group_metadata.join(measurement_summary, on="zbiralka", how="left")
        .join(change_summary, on="zbiralka", how="left")
        .with_columns(
            pl.col("n_tock_U").fill_null(0),
            pl.col("n_tock_Q").fill_null(0),
            pl.col("n_tocnih_dU_15min").fill_null(0),
            pl.col("n_tocnih_dQ_15min").fill_null(0),
            pl.col("n_razlicnih_dU").fill_null(0),
            pl.col("n_razlicnih_dQ").fill_null(0),
        )
        .sort("lokacija_od")
    )


def same_busbar_results(
    metadata: pl.DataFrame,
    correlation: np.ndarray,
    pair_counts: np.ndarray,
) -> pl.DataFrame:
    diagonal = np.diag(correlation)
    counts = np.diag(pair_counts)
    return (
        metadata.select(
            "zbiralka", "lokacija_od", "n_trafov_110_SN", "transformatorji"
        )
        .with_columns(
            pl.Series("spearman_dU_dQ", diagonal, nan_to_null=True),
            pl.Series("n_skupnih_parov_dU_dQ", counts),
        )
        .with_columns(pl.col("spearman_dU_dQ").abs().alias("abs_spearman_dU_dQ"))
        .sort("abs_spearman_dU_dQ", descending=True, nulls_last=True)
        .with_row_index("rang_po_abs_korelaciji", offset=1)
    )


def write_segments_index(path: Path, segments: list[ContinuousSegment]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["segment", "zacetek", "konec", "stevilo_casovnih_tock"])
        for segment in segments:
            writer.writerow(
                [
                    segment.tag,
                    segment.start.isoformat(sep=" "),
                    segment.end.isoformat(sep=" "),
                    segment.n_timestamps,
                ]
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
        "Spearmanova korelacija dU in dQ zbiralk 110/SN",
        fontsize=max(12, font_size * 2.2),
        color="#20252B",
        pad=18,
    )
    ax.text(
        0,
        1.012,
        "dU(t) = U(t + 15 min) - U(t); dQ(t) = Q(t + 15 min) - Q(t); "
        f"segment: {period_label}",
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
    colorbar.set_label("Spearmanov koeficient", color="#20252B")
    colorbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xlabel("dQ lokacije 110/SN [MVAr]")
    ax.set_ylabel("dU zbiralke 110 kV [kV]")
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
    busbar_values, group_metadata = load_110_sn_busbars(input_path)
    busbars = ordered_busbars(group_metadata)
    if not busbars:
        raise ValueError(
            "V vhodnih podatkih ni zbiralk 110/SN z veljavnimi meritvami U in Q."
        )

    segments = find_continuous_segments(
        busbar_values,
        max_gap_minutes=args.max_gap_minutes,
        min_segment_points=args.min_segment_points,
    )
    if not segments:
        raise ValueError("Ni dovolj dolgega zveznega segmenta za analizo.")

    write_segments_index(output_dir / "segmenti.csv", segments)
    group_metadata.write_csv(
        output_dir / "zbiralke_110_SN.csv", separator=";", include_bom=True
    )
    root_summary = [
        "SPEARMANOVA KORELACIJA dU-dQ ZBIRALK 110/SN",
        f"Vhod: {input_path}",
        f"Stevilo lokacij 110/SN: {len(busbars)}",
        f"Stevilo transformatorjev 110/SN: {group_metadata['n_trafov_110_SN'].sum()}",
        "U zbiralke: mediana veljavnih 110-kV napetosti trafov 110/SN",
        "Q lokacije: vsota Q na 110-kV straneh trafov 110/SN",
        f"Definicija: dU in dQ sta tocni {DELTA_MINUTES}-minutni spremembi",
        f"Prekinitev segmenta pri vrzeli > {args.max_gap_minutes} min",
        f"Stevilo analiziranih segmentov: {len(segments)}",
        "Vsak segment ima lastne koeficiente; segmenti se ne zdruzujejo.",
    ]
    (output_dir / "izracun_povzetek.txt").write_text(
        "\n".join(root_summary) + "\n", encoding="utf-8"
    )

    for segment in segments:
        segment_dir = output_dir / segment.tag
        segment_dir.mkdir(parents=True, exist_ok=True)
        segment_busbars = rows_in_segment(busbar_values, segment)
        changes = exact_15_minute_changes(segment_busbars)
        segment_busbar_names, segment_group_metadata = eligible_busbars_for_segment(
            changes, group_metadata, args.min_pair_points
        )
        if not segment_busbar_names:
            (segment_dir / "izracun_povzetek.txt").write_text(
                "V segmentu ni zbiralk z dovolj veljavnimi meritvami U in Q.\n",
                encoding="utf-8",
            )
            print(
                f"Segment {segment.number}/{len(segments)} preskocen: "
                "ni zbiralk z dovolj veljavnimi U in Q."
            )
            continue
        segment_busbars = segment_busbars.filter(
            pl.col("zbiralka").is_in(segment_busbar_names)
        )
        changes = changes.filter(pl.col("zbiralka").is_in(segment_busbar_names))
        du_values = wide_values(changes, segment_busbar_names, "dU_kV")
        dq_values = wide_values(changes, segment_busbar_names, "dQ_MVAr")
        correlation, pair_counts = spearman_cross_matrix(
            du_values, dq_values, args.min_pair_points
        )
        metadata = build_segment_metadata(
            segment_busbars, changes, segment_group_metadata
        )
        same_location = same_busbar_results(metadata, correlation, pair_counts)

        write_matrix_csv(
            segment_dir / "spearman_dU_dQ_korelacijska_matrika.csv",
            segment_busbar_names,
            correlation,
        )
        write_matrix_csv(
            segment_dir / "stevilo_skupnih_15min_parov_dU_dQ.csv",
            segment_busbar_names,
            pair_counts,
            integer=True,
        )
        metadata.write_csv(
            segment_dir / "zbiralke_metadata.csv", separator=";", include_bom=True
        )
        same_location.write_csv(
            segment_dir / "korelacija_dU_dQ_iste_zbiralke.csv",
            separator=";",
            include_bom=True,
        )
        draw_heatmap(
            correlation,
            segment_busbar_names,
            segment_dir / "spearman_heatmap_dU_dQ_15min.png",
            segment_dir / "spearman_heatmap_dU_dQ_15min.svg",
            args.annotate_max_busbars,
            segment.label,
        )

        finite = correlation[np.isfinite(correlation)]
        finite_same = np.diag(correlation)
        finite_same = finite_same[np.isfinite(finite_same)]
        summary_lines = [
            "SPEARMANOVA KORELACIJA dU-dQ 110/SN - ZVEZNI SEGMENT",
            f"Segment: {segment.label}",
            f"Casovnih tock segmenta: {segment.n_timestamps}",
            f"Stevilo zbiralk/lokacij z veljavnimi U in Q: {len(segment_busbar_names)}",
            f"Minimalno skupnih parov: {args.min_pair_points}",
            f"Veljavnih koeficientov v celotni matriki: {finite.size}",
            f"Veljavnih korelacij iste lokacije: {finite_same.size}",
            "Agregacija U: mediana trafov; agregacija Q: vsota trafov",
            "Manjkajoce vrednosti: brez interpolacije; parno izlocanje",
        ]
        (segment_dir / "izracun_povzetek.txt").write_text(
            "\n".join(summary_lines) + "\n", encoding="utf-8"
        )
        print(
            f"Segment {segment.number}/{len(segments)} koncan: {segment.label}"
        )

    print("Izracun vseh zveznih segmentov je koncan.")
    print(f"Zbiralk 110/SN: {len(busbars)}")
    print(f"Izhodna mapa: {output_dir}")


if __name__ == "__main__":
    main()
