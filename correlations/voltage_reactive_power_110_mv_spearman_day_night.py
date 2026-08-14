from __future__ import annotations

"""Spearmanova korelacija dU-dQ po segmentih ter po dnevnem/nočnem času.

Metodologija, izbira zbiralk 110/SN in izhodne datoteke so enaki kot v
``voltage_reactive_power_110_mv_spearman.py``. Znotraj vsakega zveznega segmenta se
izračun ponovi za tri časovna obdobja:

* vse meritve;
* dnevne meritve: 05:00 <= t < 20:00;
* nočne meritve: 20:00 <= t ali t < 05:00.

Časovno obdobje določa začetni čas ``t`` točnega 15-minutnega para
``t -> t + 15 min``. Manjkajoče vrednosti se ne interpolirajo, meritve iz
različnih zveznih segmentov pa se ne združujejo.

Privzeti zagon iz korena projekta:

    python -m correlations.voltage_reactive_power_110_mv_spearman_day_night
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

try:
    from . import voltage_reactive_power_110_mv_spearman as base
except ImportError:  # Support direct execution from this directory.
    import voltage_reactive_power_110_mv_spearman as base


DAY_START_HOUR = 5
NIGHT_START_HOUR = 20


@dataclass(frozen=True)
class TimePeriod:
    directory: str
    label: str
    mode: str


TIME_PERIODS = (
    TimePeriod("vse_meritve", "all measurements", "all"),
    TimePeriod("dnevne_meritve", "daytime measurements (05:00-20:00)", "day"),
    TimePeriod("nocne_meritve", "nighttime measurements (20:00-05:00)", "night"),
)


def default_output_dir() -> Path:
    return (
        base.project_root()
        / "Uman meritve"
        / "korelacija_U_Q_110_SN_spearman_dan_noc"
        / "zvezni_segmenti"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Izračuna Spearmanovo korelacijo točnih 15-minutnih sprememb "
            "U in Q po zveznih segmentih, ločeno za vse, dnevne in nočne meritve."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=base.default_input_path(),
        help="Vhodni transformers_wide.parquet.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir(),
        help="Izhodna mapa za segmente ter dnevne/nočne rezultate.",
    )
    parser.add_argument(
        "--min-pair-points",
        type=int,
        default=base.DEFAULT_MIN_PAIR_POINTS,
        help="Najmanj skupnih veljavnih parov dU-dQ (privzeto: 20).",
    )
    parser.add_argument(
        "--max-gap-minutes",
        type=int,
        default=base.DEFAULT_MAX_GAP_MINUTES,
        help="Največji razmik znotraj zveznega segmenta (privzeto: 15 min).",
    )
    parser.add_argument(
        "--min-segment-points",
        type=int,
        default=base.DEFAULT_MIN_SEGMENT_POINTS,
        help="Najmanj časovnih točk analiziranega segmenta (privzeto: 20).",
    )
    parser.add_argument(
        "--annotate-max-busbars",
        type=int,
        default=45,
        help="Največje število zbiralk za koeficiente v celicah heatmap.",
    )
    return parser.parse_args()


def rows_in_time_period(frame: pl.DataFrame, period: TimePeriod) -> pl.DataFrame:
    """Filtrira vrstice glede na uro začetnega časa meritve oziroma para."""
    if period.mode == "all":
        return frame

    hour = pl.col("time").dt.hour()
    is_day = (hour >= DAY_START_HOUR) & (hour < NIGHT_START_HOUR)
    if period.mode == "day":
        return frame.filter(is_day)
    if period.mode == "night":
        return frame.filter(~is_day)
    raise ValueError(f"Neznan način časovnega obdobja: {period.mode}")


def analyse_period(
    period_dir: Path,
    period: TimePeriod,
    segment: base.ContinuousSegment,
    segment_busbars: pl.DataFrame,
    all_changes: pl.DataFrame,
    group_metadata: pl.DataFrame,
    min_pair_points: int,
    annotate_max_busbars: int,
) -> tuple[int, int]:
    """Izvede in shrani analizo enega obdobja; vrne (zbiralke, pari)."""
    period_dir.mkdir(parents=True, exist_ok=True)
    period_changes = rows_in_time_period(all_changes, period)
    period_measurements = rows_in_time_period(segment_busbars, period)
    names, selected_metadata = base.eligible_busbars_for_segment(
        period_changes, group_metadata, min_pair_points
    )

    if not names:
        (period_dir / "izracun_povzetek.txt").write_text(
            "V obdobju ni zbiralk z dovolj veljavnimi dnevnimi/nočnimi "
            "meritvami U in Q.\n",
            encoding="utf-8",
        )
        return 0, period_changes.select("time").n_unique()

    period_changes = period_changes.filter(pl.col("zbiralka").is_in(names))
    period_measurements = period_measurements.filter(
        pl.col("zbiralka").is_in(names)
    )
    du_values = base.wide_values(period_changes, names, "dU_kV")
    dq_values = base.wide_values(period_changes, names, "dQ_MVAr")
    correlation, pair_counts = base.spearman_cross_matrix(
        du_values, dq_values, min_pair_points
    )
    metadata = base.build_segment_metadata(
        period_measurements, period_changes, selected_metadata
    )
    same_location = base.same_busbar_results(metadata, correlation, pair_counts)

    base.write_matrix_csv(
        period_dir / "spearman_dU_dQ_korelacijska_matrika.csv",
        names,
        correlation,
    )
    base.write_matrix_csv(
        period_dir / "stevilo_skupnih_15min_parov_dU_dQ.csv",
        names,
        pair_counts,
        integer=True,
    )
    metadata.write_csv(
        period_dir / "zbiralke_metadata.csv", separator=";", include_bom=True
    )
    same_location.write_csv(
        period_dir / "korelacija_dU_dQ_iste_zbiralke.csv",
        separator=";",
        include_bom=True,
    )
    base.draw_heatmap(
        correlation,
        names,
        period_dir / "spearman_heatmap_dU_dQ_15min.png",
        period_dir / "spearman_heatmap_dU_dQ_15min.svg",
        annotate_max_busbars,
        f"{segment.label}; {period.label}",
    )

    finite = correlation[np.isfinite(correlation)]
    finite_same = np.diag(correlation)
    finite_same = finite_same[np.isfinite(finite_same)]
    n_times = period_changes.select("time").n_unique()
    summary = [
        "SPEARMANOVA KORELACIJA dU-dQ 110/SN - ČASOVNO OBDOBJE",
        f"Segment: {segment.label}",
        f"Obdobje: {period.label}",
        "Razvrstitev obdobja temelji na začetnem času t 15-minutnega para.",
        f"Časovnih točk s spremembami v obdobju: {n_times}",
        f"Število zbiralk/lokacij z veljavnimi U in Q: {len(names)}",
        f"Minimalno skupnih parov: {min_pair_points}",
        f"Veljavnih koeficientov v celotni matriki: {finite.size}",
        f"Veljavnih korelacij iste lokacije: {finite_same.size}",
        "Agregacija U: mediana trafov; agregacija Q: vsota trafov",
        "Manjkajoče vrednosti: brez interpolacije; parno izločanje",
    ]
    (period_dir / "izracun_povzetek.txt").write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )
    return len(names), n_times


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

    base.validate_schema(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    busbar_values, group_metadata = base.load_110_sn_busbars(input_path)
    busbars = base.ordered_busbars(group_metadata)
    if not busbars:
        raise ValueError(
            "V vhodnih podatkih ni zbiralk 110/SN z veljavnimi meritvami U in Q."
        )

    segments = base.find_continuous_segments(
        busbar_values,
        max_gap_minutes=args.max_gap_minutes,
        min_segment_points=args.min_segment_points,
    )
    if not segments:
        raise ValueError("Ni dovolj dolgega zveznega segmenta za analizo.")

    base.write_segments_index(output_dir / "segmenti.csv", segments)
    group_metadata.write_csv(
        output_dir / "zbiralke_110_SN.csv", separator=";", include_bom=True
    )
    root_summary = [
        "SPEARMANOVA KORELACIJA dU-dQ ZBIRALK 110/SN - DAN IN NOČ",
        f"Vhod: {input_path}",
        f"Število lokacij 110/SN: {len(busbars)}",
        f"Število analiziranih segmentov: {len(segments)}",
        "Dan: 05:00 <= t < 20:00",
        "Noč: 20:00 <= t ali t < 05:00",
        "Obdobje določa začetni čas t točnega 15-minutnega para.",
        "Vsak segment in časovno obdobje imata lastne koeficiente.",
    ]
    (output_dir / "izracun_povzetek.txt").write_text(
        "\n".join(root_summary) + "\n", encoding="utf-8"
    )

    overview_rows: list[dict[str, object]] = []
    for segment in segments:
        segment_dir = output_dir / segment.tag
        segment_dir.mkdir(parents=True, exist_ok=True)
        segment_busbars = base.rows_in_segment(busbar_values, segment)
        all_changes = base.exact_15_minute_changes(segment_busbars)

        for period in TIME_PERIODS:
            n_busbars, n_times = analyse_period(
                segment_dir / period.directory,
                period,
                segment,
                segment_busbars,
                all_changes,
                group_metadata,
                args.min_pair_points,
                args.annotate_max_busbars,
            )
            overview_rows.append(
                {
                    "segment": segment.tag,
                    "zacetek": segment.start,
                    "konec": segment.end,
                    "obdobje": period.directory,
                    "opis_obdobja": period.label,
                    "n_casovnih_tock_sprememb": n_times,
                    "n_ustreznih_zbiralk": n_busbars,
                }
            )
        print(f"Segment {segment.number}/{len(segments)} končan: {segment.label}")

    pl.DataFrame(overview_rows).write_csv(
        output_dir / "pregled_segmentov_in_obdobij.csv",
        separator=";",
        include_bom=True,
    )
    print("Izračun vseh segmentov ter dnevnih/nočnih obdobij je končan.")
    print(f"Zbiralk 110/SN: {len(busbars)}")
    print(f"Izhodna mapa: {output_dir}")


if __name__ == "__main__":
    main()
