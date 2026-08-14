from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .reactive_power_peak import build_110_mv_candidates, build_q_matrix
except ImportError:  # Support direct execution from this directory.
    from reactive_power_peak import build_110_mv_candidates, build_q_matrix


DEFAULT_YEAR = 2025
DEFAULT_MIN_COVERAGE = 0.95
DEFAULT_INTERVAL_MINUTES = 15

COLORS = {
    "curve": "#2463A8",
    "zero": "#343A40",
    "grid": "#D9DEE7",
}


def default_component_dir() -> Path:
    return (
        Path(__file__).resolve().parent.parent
        / "Uman meritve"
        / "Pridobljeno in urejeno"
        / "urejeno"
        / "Uman_parquet"
        / "component_files"
    )


def default_output_dir() -> Path:
    return (
        Path(__file__).resolve().parent.parent
        / "Uman meritve"
        / "letni_urejeni_diagrami_jalove_moci"
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sesteje pretoke jalove moci vseh transformatorjev 110/SN in "
            "izrise urejeni diagram skupne jalove moci za koledarsko leto."
        )
    )
    parser.add_argument(
        "--component-dir",
        type=Path,
        default=default_component_dir(),
        help="Mapa s komponentnimi datotekami TR_*.parquet.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir(),
        help="Izhodna mapa za SVG in CSV datoteki.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=DEFAULT_YEAR,
        help=f"Koledarsko leto analize (privzeto: {DEFAULT_YEAR}).",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=DEFAULT_MIN_COVERAGE,
        help=(
            "Najmanjsi zahtevani delez razpolozljivih meritev "
            f"transformatorjev (privzeto: {DEFAULT_MIN_COVERAGE:.0%}%)."
        ),
    )
    parser.add_argument(
        "--interval-minutes",
        type=int,
        default=DEFAULT_INTERVAL_MINUTES,
        help=(
            "Casovni korak, na katerega se meritve pred sestevanjem "
            f"povprecijo (privzeto: {DEFAULT_INTERVAL_MINUTES} min)."
        ),
    )
    args = parser.parse_args()
    args.component_dir = args.component_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()

    if not args.component_dir.is_dir():
        parser.error(f"Vhodna mapa ne obstaja: {args.component_dir}")
    if not 1900 <= args.year <= 2200:
        parser.error("--year mora biti med 1900 in 2200.")
    if not 0 <= args.min_coverage <= 1:
        parser.error("--min-coverage mora biti med 0 in 1.")
    if args.interval_minutes < 1:
        parser.error("--interval-minutes mora biti vsaj 1.")
    return args


def expected_measurement_count(year: int, interval_minutes: int) -> int:
    start = datetime(year, 1, 1)
    end = datetime(year + 1, 1, 1)
    return int((end - start).total_seconds() // (interval_minutes * 60))


def calculate_annual_total_q(
    q_matrix: pd.DataFrame,
    year: int,
    interval_minutes: int,
    min_coverage: float,
) -> tuple[pd.DataFrame, int]:
    """Return valid annual system-Q values and the expected annual count."""
    start = pd.Timestamp(year=year, month=1, day=1)
    end = pd.Timestamp(year=year + 1, month=1, day=1)
    annual = q_matrix.loc[(q_matrix.index >= start) & (q_matrix.index < end)]
    if annual.empty:
        raise RuntimeError(f"Za leto {year} ni meritev jalove moci.")

    # More than one raw value in the same interval is averaged per transformer.
    annual = annual.resample(f"{interval_minutes}min").mean()
    transformer_count = annual.shape[1]
    available_count = annual.notna().sum(axis=1)
    coverage = available_count / transformer_count
    total_q = annual.sum(axis=1, min_count=1)

    valid = total_q.notna() & (coverage >= min_coverage)
    result = pd.DataFrame(
        {
            "time": annual.index[valid],
            "total_Q_MVAr": total_q.loc[valid].to_numpy(),
            "available_transformers": available_count.loc[valid].to_numpy(),
            "coverage_pct": coverage.loc[valid].to_numpy() * 100.0,
        }
    )
    if result.empty:
        raise RuntimeError(
            "Noben casovni interval ne dosega zahtevane pokritosti meritev "
            f"({min_coverage:.1%})."
        )

    return result, expected_measurement_count(year, interval_minutes)


def build_duration_curve(
    annual_total_q: pd.DataFrame,
) -> pd.DataFrame:
    """Sort total Q from the most capacitive to the most inductive value."""
    ordered_q = np.sort(annual_total_q["total_Q_MVAr"].to_numpy(dtype=float))
    if len(ordered_q) == 1:
        percent = np.array([0.0])
    else:
        percent = np.linspace(0.0, 100.0, len(ordered_q))
    return pd.DataFrame(
        {
            "percent_of_valid_year": percent,
            "total_Q_MVAr": ordered_q,
        }
    )


def plot_duration_curve(
    curve: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12.0, 7.0), constrained_layout=True)
    ax.plot(
        curve["percent_of_valid_year"],
        curve["total_Q_MVAr"],
        color=COLORS["curve"],
        linewidth=2.2,
    )
    ax.axhline(0.0, color=COLORS["zero"], linewidth=1.0, linestyle="--")
    ax.set_xlabel("Percentage of valid annual measurements / %")
    ax.set_ylabel("Q / MVAr")
    ax.set_xlim(0.0, 100.0)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8, alpha=0.8)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.5, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#6C7480")
    ax.spines["bottom"].set_color("#6C7480")

    fig.savefig(output_path, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_arguments()
    all_files = sorted(args.component_dir.glob("TR_*.parquet"))
    if not all_files:
        raise FileNotFoundError(
            f"V mapi ni datotek TR_*.parquet: {args.component_dir}"
        )

    candidates, excluded_count = build_110_mv_candidates(all_files)
    if not candidates:
        raise RuntimeError("Ni veljavnih parov transformatorjev 110/SN.")

    print(f"Izbranih datotek transformatorjev 110/SN: {len(candidates)}")
    print(f"Izlocenih neustreznih 110 kV datotek: {excluded_count}")
    print("Berem in sestevam meritve jalove moci ...")
    q_matrix, column_meta = build_q_matrix(candidates)
    if q_matrix.empty:
        raise RuntimeError("Ni veljavnih meritev jalove moci.")

    annual_total_q, expected_count = calculate_annual_total_q(
        q_matrix=q_matrix,
        year=args.year,
        interval_minutes=args.interval_minutes,
        min_coverage=args.min_coverage,
    )
    curve = build_duration_curve(annual_total_q)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"letni_urejeni_diagram_jalove_moci_{args.year}"
    svg_path = args.output_dir / f"{stem}.svg"
    curve_csv_path = args.output_dir / f"{stem}.csv"
    time_series_csv_path = args.output_dir / f"vsota_jalove_moci_{args.year}.csv"

    plot_duration_curve(
        curve=curve,
        output_path=svg_path,
    )
    curve.to_csv(curve_csv_path, index=False, sep=";", decimal=",")
    annual_total_q.to_csv(
        time_series_csv_path, index=False, sep=";", decimal=","
    )

    annual_coverage = 100.0 * len(annual_total_q) / expected_count
    print("\nLETNI UREJENI DIAGRAM SKUPNE JALOVE MOCI")
    print("=" * 78)
    print(f"Leto:                     {args.year}")
    print(f"Transformatorjev:         {len(column_meta)}")
    print(f"Veljavnih intervalov:     {len(annual_total_q)}/{expected_count}")
    print(f"Pokritost leta:           {annual_coverage:.2f} %")
    print(f"Najbolj kapacitivna Q:    {curve['total_Q_MVAr'].min():.3f} MVAr")
    print(f"Najbolj induktivna Q:     {curve['total_Q_MVAr'].max():.3f} MVAr")
    print(f"SVG diagram:              {svg_path}")
    print(f"CSV urejena krivulja:     {curve_csv_path}")
    print(f"CSV casovna vrsta:        {time_series_csv_path}")


if __name__ == "__main__":
    main()
