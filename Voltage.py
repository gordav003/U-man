from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import polars as pl


HV_LEVELS_KV = (110, 220, 400)
DEFAULT_THRESHOLDS_PU = {
    110: 1.12,
    220: 1.11,
    400: 1.05,
}


def default_input_path() -> Path:
    project_root = Path(__file__).resolve().parent.parent
    return (
        project_root
        / "Uman meritve"
        / "2026_06_17  SCADA meritve 4600"
        / "urejeno"
        / "Uman_parquet"
        / "transformers_wide.parquet"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Najde trenutke z napetostmi nad pragom na 110-, 220- in 400-kV "
            "straneh transformatorjev. Rezultat izpise samo v konzolo."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input_path(),
        help="Pot do transformers_wide.parquet.",
    )
    parser.add_argument(
        "--threshold-pu",
        type=float,
        default=None,
        help=(
            "Po zelji preglasi vse tri privzete pragove z enim skupnim pragom."
        ),
    )
    parser.add_argument(
        "--threshold-110-pu",
        type=float,
        default=None,
        help="Po zelji preglasi privzeti prag 1.12 pu za nivo 110 kV.",
    )
    parser.add_argument(
        "--threshold-220-pu",
        type=float,
        default=None,
        help="Po zelji preglasi privzeti prag 1.11 pu za nivo 220 kV.",
    )
    parser.add_argument(
        "--threshold-400-pu",
        type=float,
        default=None,
        help="Po zelji preglasi privzeti prag 1.05 pu za nivo 400 kV.",
    )
    parser.add_argument(
        "--moments",
        type=int,
        default=3,
        help="Stevilo izbranih trenutkov (privzeto: 3).",
    )
    parser.add_argument(
        "--min-separation-hours",
        type=float,
        default=24.0,
        help=(
            "Najmanjsi razmik med izbranimi trenutki, da zaporedni 15-minutni "
            "vzorci istega dogodka niso izbrani veckrat (privzeto: 24 h)."
        ),
    )
    parser.add_argument(
        "--rank-by",
        choices=("count", "max"),
        default="count",
        help=(
            "'count': najprej najvec RTP-jev nad pragom; "
            "'max': najprej najvisja posamezna napetost (privzeto: count)."
        ),
    )
    parser.add_argument(
        "--top-rtps",
        type=int,
        default=10,
        help=(
            "Najvecje stevilo različnih RTP-jev, izpisanih pri vsakem "
            "trenutku (privzeto: 10)."
        ),
    )
    parser.add_argument(
        "--only-110",
        action="store_true",
        help=(
            "Analizo in izpis omeji samo na transformatorje na 110-kV nivoju. "
            "Brez te opcije se uporabijo nivoji 110, 220 in 400 kV."
        ),
    )
    return parser.parse_args()


def transformer_rows(
    parquet_path: Path,
    voltage_levels_kv: tuple[int, ...],
) -> pl.LazyFrame:
    required = {
        "time",
        "component_id",
        "napetost_kv",
        "lokacija_od",
        "objekt",
        "U",
    }

    lazy = pl.scan_parquet(parquet_path)
    available = set(lazy.collect_schema().names())
    missing = sorted(required - available)
    if missing:
        raise ValueError(
            "V vhodnem Parquetu manjkajo stolpci: " + ", ".join(missing)
        )

    # U je v urejenih SCADA podatkih podan v kV. Grupiranje zagotovi, da se isti
    # transformator v istem trenutku ne steje veckrat, ce obstaja vec zapisov.
    return (
        lazy.filter(pl.col("napetost_kv").is_in(voltage_levels_kv))
        .filter(pl.col("U").is_not_null())
        .group_by(
            ["time", "component_id", "napetost_kv", "lokacija_od", "objekt"]
        )
        .agg(pl.col("U").max().alias("U_kV"))
        .with_columns(
            (pl.col("U_kV") / pl.col("napetost_kv")).alias("U_pu")
        )
        # Izloci nicelne meritve in ocitne napake/enote, ki niso kV.
        .filter(pl.col("U_pu").is_between(0.5, 1.5, closed="both"))
    )


def choose_separated_moments(
    candidates: pl.DataFrame,
    number_of_moments: int,
    minimum_separation: timedelta,
) -> list:
    selected = []

    for moment in candidates.get_column("time").to_list():
        if all(abs(moment - previous) >= minimum_separation for previous in selected):
            selected.append(moment)
            if len(selected) == number_of_moments:
                break

    return selected


def format_transformer(row: dict) -> str:
    location = row.get("lokacija_od") or "?"
    obj = row.get("objekt") or row["component_id"]
    return f"{location} {obj}"


def build_moment_summary(
    rows: pl.LazyFrame,
    threshold_pu: float,
    rank_by: str,
) -> pl.LazyFrame:
    above_threshold = pl.col("U_pu") > threshold_pu

    summary = (
        rows.group_by("time")
        .agg(
            above_threshold.sum().alias("n_transformers_above"),
            pl.col("lokacija_od")
            .filter(above_threshold & pl.col("lokacija_od").is_not_null())
            .n_unique()
            .alias("n_rtps_above"),
            pl.col("U_pu").max().alias("max_U_pu"),
            pl.len().alias("n_valid"),
        )
        .filter(pl.col("n_transformers_above") > 0)
    )

    if rank_by == "count":
        return summary.sort(
            ["n_rtps_above", "n_transformers_above", "max_U_pu", "time"],
            descending=[True, True, True, False],
        )

    return summary.sort(
        ["max_U_pu", "n_rtps_above", "n_transformers_above", "time"],
        descending=[True, True, True, False],
    )


def top_distinct_rtps(above: pl.DataFrame, limit: int) -> list[dict]:
    """Return the highest-voltage transformer for each of the top RTPs."""
    result = []
    seen_rtps = set()

    for row in above.sort("U_pu", descending=True).iter_rows(named=True):
        rtp = row.get("lokacija_od") or "?"
        if rtp in seen_rtps:
            continue

        seen_rtps.add(rtp)
        result.append(row)
        if len(result) == limit:
            break

    return result


def print_level_results(
    rows: pl.LazyFrame,
    voltage_level_kv: int,
    threshold_pu: float,
    number_of_moments: int,
    minimum_separation: timedelta,
    rank_by: str,
    top_rtps_limit: int,
) -> None:
    level_rows = rows.filter(pl.col("napetost_kv") == voltage_level_kv)
    candidates = build_moment_summary(
        level_rows,
        threshold_pu=threshold_pu,
        rank_by=rank_by,
    ).collect()

    print("\n" + "#" * 88)
    print(f"NAPETOSTNI NIVO: {voltage_level_kv} kV")
    print("#" * 88)

    if candidates.is_empty():
        print(
            f"Ni trenutkov, v katerih bi bil kateri od transformatorjev "
            f"na nivoju {voltage_level_kv} kV nad {threshold_pu:.4f} pu."
        )
        return

    selected_times = choose_separated_moments(
        candidates,
        number_of_moments=number_of_moments,
        minimum_separation=minimum_separation,
    )

    if len(selected_times) < number_of_moments:
        print(
            f"OPOZORILO: z zahtevanim razmikom je bilo mogoce izbrati samo "
            f"{len(selected_times)} od {number_of_moments} trenutkov."
        )

    details = (
        level_rows.filter(pl.col("time").is_in(selected_times))
        .sort(["time", "U_pu"], descending=[False, True])
        .collect()
    )

    for index, moment in enumerate(selected_times, start=1):
        moment_rows = details.filter(pl.col("time") == moment)
        above = moment_rows.filter(pl.col("U_pu") > threshold_pu)
        top_rtps = top_distinct_rtps(above, top_rtps_limit)
        maximum_pu = moment_rows.get_column("U_pu").max()
        highest = moment_rows.filter(
            (pl.col("U_pu") - maximum_pu).abs() < 1e-9
        )

        print(f"\n{index}. TRENUTEK: {moment}")
        print(
            f"   RTP-jev nad pragom: "
            f"{above.get_column('lokacija_od').drop_nulls().n_unique()}"
        )
        print(f"   Transformatorjev nad pragom: {above.height}")
        print(f"   Veljavnih meritev na nivoju: {moment_rows.height}")
        print("   Najvisja napetost v trenutku:")
        for row in highest.iter_rows(named=True):
            print(
                f"     - {format_transformer(row)} | "
                f"U = {row['U_kV']:.3f} kV | U = {row['U_pu']:.5f} pu"
            )

        print(
            f"   TOP {min(top_rtps_limit, len(top_rtps))} RTP-jev "
            f"nad {threshold_pu:.4f} pu:"
        )
        for rank, row in enumerate(top_rtps, start=1):
            rtp = row.get("lokacija_od") or "?"
            transformer = row.get("objekt") or row["component_id"]
            print(
                f"     {rank:>2}. {rtp:20s} | trafo {transformer:12s} | "
                f"U = {row['U_kV']:>8.3f} kV | U = {row['U_pu']:.5f} pu"
            )


def main() -> None:
    args = parse_args()
    parquet_path = args.input.resolve()
    voltage_levels_kv = (110,) if args.only_110 else HV_LEVELS_KV
    thresholds_pu = DEFAULT_THRESHOLDS_PU.copy()

    if args.threshold_pu is not None:
        thresholds_pu = {
            voltage_level_kv: args.threshold_pu
            for voltage_level_kv in HV_LEVELS_KV
        }

    level_overrides = {
        110: args.threshold_110_pu,
        220: args.threshold_220_pu,
        400: args.threshold_400_pu,
    }
    for voltage_level_kv, override in level_overrides.items():
        if override is not None:
            thresholds_pu[voltage_level_kv] = override

    if not parquet_path.is_file():
        raise FileNotFoundError(f"Vhodna datoteka ne obstaja: {parquet_path}")
    if args.moments < 1:
        raise ValueError("--moments mora biti vsaj 1.")
    if args.min_separation_hours < 0:
        raise ValueError("--min-separation-hours ne sme biti negativen.")
    if args.top_rtps < 1:
        raise ValueError("--top-rtps mora biti vsaj 1.")
    if any(threshold <= 0 for threshold in thresholds_pu.values()):
        raise ValueError("Vsi napetostni pragovi morajo biti vecji od 0 pu.")

    rows = transformer_rows(parquet_path, voltage_levels_kv)

    print("=" * 88)
    print("VISOKE VN NAPETOSTI TRANSFORMATORJEV")
    print(f"Vhod: {parquet_path}")
    print(f"VN nivoji: {', '.join(map(str, voltage_levels_kv))} kV")
    print(
        "Pragovi: "
        + "; ".join(
            f"{voltage_level_kv} kV > "
            f"{thresholds_pu[voltage_level_kv]:.4f} pu"
            for voltage_level_kv in voltage_levels_kv
        )
    )
    print(f"Razvrscanje: {args.rank_by}")
    print(f"Najvec RTP-jev v posameznem izpisu: {args.top_rtps}")
    print(f"Najmanjsi razmik med dogodki: {args.min_separation_hours:g} h")
    print("=" * 88)

    for voltage_level_kv in voltage_levels_kv:
        print_level_results(
            rows=rows,
            voltage_level_kv=voltage_level_kv,
            threshold_pu=thresholds_pu[voltage_level_kv],
            number_of_moments=args.moments,
            minimum_separation=timedelta(hours=args.min_separation_hours),
            rank_by=args.rank_by,
            top_rtps_limit=args.top_rtps,
        )

    print("\n" + "=" * 88)
    print("Konec. Skripta ni ustvarila nobene izhodne datoteke.")


if __name__ == "__main__":
    main()
