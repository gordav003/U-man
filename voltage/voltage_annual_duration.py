from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

try:
    from .voltage_data import (
        TransformerFile,
        default_component_dir,
        discover_transformers,
    )
except ImportError:  # Support direct execution from this directory.
    from voltage_data import (
        TransformerFile,
        default_component_dir,
        discover_transformers,
    )


VOLTAGE_LEVELS_KV = (400, 220, 110)
DEFAULT_YEAR = 2025
DEFAULT_CURVE_POINTS = 501
DEFAULT_VALID_MIN_PU = 0.8
DEFAULT_VALID_MAX_PU = 1.2
DEFAULT_MAX_MEDIAN_DEVIATION_PU = 0.15

COLORS = {
    "aggregate": "#2463A8",
    "grid": "#D9DEE7",
}


@dataclass(frozen=True)
class TransformerCurve:
    transformer: TransformerFile
    values_kv: np.ndarray
    first_time: datetime
    last_time: datetime
    unique_timestamps: int
    coverage_percent: float


def default_output_dir() -> Path:
    return (
        Path(__file__).resolve().parent.parent
        / "Uman meritve"
        / "letni_urejeni_diagrami_napetosti"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Izrise letne urejene diagrame napetosti posebej za 400, 220 "
            "in 110 kV iz posameznih Parquet datotek transformatorjev."
        )
    )
    parser.add_argument(
        "--component-dir",
        type=Path,
        default=default_component_dir(),
        help="Mapa s TR_*.parquet datotekami.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir(),
        help="Izhodna mapa za SVG datoteke.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=DEFAULT_YEAR,
        help=f"Koledarsko leto analize (privzeto: {DEFAULT_YEAR}).",
    )
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        choices=VOLTAGE_LEVELS_KV,
        default=list(VOLTAGE_LEVELS_KV),
        help="Napetostni nivoji za izris (privzeto: 400 220 110).",
    )
    parser.add_argument(
        "--unit",
        choices=("kv", "pu"),
        default="kv",
        help="Enota na navpicni osi (privzeto: kv).",
    )
    parser.add_argument(
        "--curve-points",
        type=int,
        default=DEFAULT_CURVE_POINTS,
        help=(
            "Stevilo enakomerno razporejenih tock na osi trajanja "
            f"(privzeto: {DEFAULT_CURVE_POINTS})."
        ),
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=4,
        help="Najmanjse stevilo veljavnih meritev za vkljucitev transformatorja.",
    )
    parser.add_argument(
        "--valid-min-pu",
        type=float,
        default=DEFAULT_VALID_MIN_PU,
        help=(
            "Spodnja meja veljavne napetosti v p.u. "
            f"(privzeto: {DEFAULT_VALID_MIN_PU})."
        ),
    )
    parser.add_argument(
        "--valid-max-pu",
        type=float,
        default=DEFAULT_VALID_MAX_PU,
        help=(
            "Zgornja meja veljavne napetosti v p.u. "
            f"(privzeto: {DEFAULT_VALID_MAX_PU})."
        ),
    )
    parser.add_argument(
        "--max-median-deviation-pu",
        type=float,
        default=DEFAULT_MAX_MEDIAN_DEVIATION_PU,
        help=(
            "Najvecje dovoljeno odstopanje od letne mediane posameznega "
            "transformatorja, izraženo glede na nazivni nivo; 0 izklopi "
            f"ta filter (privzeto: {DEFAULT_MAX_MEDIAN_DEVIATION_PU})."
        ),
    )
    return parser.parse_args()


def expected_quarter_hours(year: int) -> int:
    start = datetime(year, 1, 1)
    end = datetime(year + 1, 1, 1)
    return int((end - start).total_seconds() // (15 * 60))


def read_transformer_curve(
    transformer: TransformerFile,
    year: int,
    valid_min_pu: float,
    valid_max_pu: float,
    max_median_deviation_pu: float,
) -> TransformerCurve | None:
    start = datetime(year, 1, 1)
    end = datetime(year + 1, 1, 1)
    schema = pl.scan_parquet(transformer.path).collect_schema()
    missing = {"time", "U"} - set(schema.names())
    if missing:
        raise ValueError(
            f"manjkajo stolpci: {', '.join(sorted(missing))}"
        )

    # Isti casovni zapis se lahko v vhodu pojavi veckrat. Pred urejanjem ga
    # zdruzimo v eno povprecno vrednost, da podvojeni zapisi nimajo dodatne teze.
    data = (
        pl.scan_parquet(transformer.path)
        .filter(
            (pl.col("time") >= start)
            & (pl.col("time") < end)
            & pl.col("U").is_not_null()
            & pl.col("U").is_finite()
            & pl.col("U").is_between(
                transformer.voltage_kv * valid_min_pu,
                transformer.voltage_kv * valid_max_pu,
                closed="both",
            )
        )
        .group_by("time")
        .agg(pl.col("U").mean().alias("U"))
        .sort("time")
        .collect()
    )
    if data.is_empty():
        return None

    # Posamezne SCADA tocke imajo v nekaterih datotekah ocitno napacno skalo
    # (npr. priblizno polovico nazivne napetosti). Robustni filter glede na
    # mediano jih odstrani, ne da bi gladil ali spreminjal preostale meritve.
    if max_median_deviation_pu > 0:
        median_kv = data.get_column("U").median()
        maximum_deviation_kv = (
            transformer.voltage_kv * max_median_deviation_pu
        )
        data = data.filter(
            (pl.col("U") - median_kv).abs() <= maximum_deviation_kv
        )
        if data.is_empty():
            return None

    values = data.get_column("U").to_numpy()
    unique_timestamps = len(values)
    return TransformerCurve(
        transformer=transformer,
        values_kv=values,
        first_time=data.item(0, "time"),
        last_time=data.item(-1, "time"),
        unique_timestamps=unique_timestamps,
        coverage_percent=(
            100.0 * unique_timestamps / expected_quarter_hours(year)
        ),
    )


def duration_curve(values: np.ndarray, x_percent: np.ndarray) -> np.ndarray:
    """Vrne padajoce urejene vrednosti na skupni odstotni osi trajanja."""
    ordered = np.sort(np.asarray(values, dtype=float))[::-1]
    source_x = np.linspace(0.0, 100.0, ordered.size)
    return np.interp(x_percent, source_x, ordered)


def convert_unit(values_kv: np.ndarray, level_kv: int, unit: str) -> np.ndarray:
    if unit == "pu":
        return values_kv / level_kv
    return values_kv


def transformer_label(curve: TransformerCurve) -> str:
    item = curve.transformer
    return f"{item.station} {item.transformer}"


def plot_level(
    curves: list[TransformerCurve],
    level_kv: int,
    year: int,
    unit: str,
    curve_points: int,
    output_dir: Path,
) -> Path:
    x_percent = np.linspace(0.0, 100.0, curve_points)
    all_values = np.concatenate(
        [convert_unit(curve.values_kv, level_kv, unit) for curve in curves]
    )
    aggregate = duration_curve(all_values, x_percent)

    fig, ax = plt.subplots(figsize=(12.0, 7.0), constrained_layout=True)
    ax.plot(
        x_percent,
        aggregate,
        color=COLORS["aggregate"],
        linewidth=2.4,
        label="All valid measurements",
        zorder=4,
    )

    unit_label = "kV" if unit == "kv" else "p.u."
    ax.set_xlabel(
        "Share of valid measurements above the displayed voltage / %"
    )
    ax.set_ylabel(f"U / {unit_label}")
    ax.set_xlim(0.0, 100.0)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8, alpha=0.8)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.5, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#6C7480")
    ax.spines["bottom"].set_color("#6C7480")
    ax.legend(loc="best", frameon=False)

    stem = f"letni_urejeni_diagram_napetosti_{level_kv}kV_{year}_{unit}"
    svg_path = output_dir / f"{stem}.svg"
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return svg_path


def main() -> None:
    args = parse_args()
    component_dir = args.component_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not component_dir.is_dir():
        raise FileNotFoundError(f"Vhodna mapa ne obstaja: {component_dir}")
    if not 1900 <= args.year <= 2200:
        raise ValueError("--year mora biti med 1900 in 2200.")
    if args.curve_points < 2:
        raise ValueError("--curve-points mora biti vsaj 2.")
    if args.min_points < 2:
        raise ValueError("--min-points mora biti vsaj 2.")
    if not 0 < args.valid_min_pu < args.valid_max_pu:
        raise ValueError(
            "Veljati mora 0 < --valid-min-pu < --valid-max-pu."
        )
    if args.max_median_deviation_pu < 0:
        raise ValueError("--max-median-deviation-pu ne sme biti negativen.")

    requested_levels = tuple(dict.fromkeys(args.levels))
    transformers = [
        item
        for item in discover_transformers(component_dir)
        if item.voltage_kv in requested_levels
    ]
    if not transformers:
        raise ValueError(
            "V vhodni mapi ni transformatorjev za zahtevane napetostne nivoje."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    curves_by_level: dict[int, list[TransformerCurve]] = {
        level: [] for level in requested_levels
    }
    warnings: list[str] = []

    print("=" * 88)
    print("LETNI UREJENI DIAGRAMI NAPETOSTI")
    print(f"Vhod:  {component_dir}")
    print(f"Izhod: {output_dir}")
    print(f"Leto:  {args.year}")
    print("=" * 88)

    for index, transformer in enumerate(transformers, start=1):
        try:
            curve = read_transformer_curve(
                transformer,
                args.year,
                args.valid_min_pu,
                args.valid_max_pu,
                args.max_median_deviation_pu,
            )
        except Exception as error:
            warnings.append(f"{transformer.path.name}: {error}")
            continue

        if curve is None:
            continue
        if curve.unique_timestamps < args.min_points:
            warnings.append(
                f"{transformer.path.name}: samo {curve.unique_timestamps} "
                "veljavnih meritev"
            )
            continue
        curves_by_level[transformer.voltage_kv].append(curve)

        if index % 50 == 0 or index == len(transformers):
            print(f"Prebranih datotek: {index}/{len(transformers)}")

    created: list[Path] = []
    for level_kv in requested_levels:
        curves = curves_by_level[level_kv]
        if not curves:
            warnings.append(
                f"{level_kv} kV: v letu {args.year} ni dovolj veljavnih meritev"
            )
            continue

        output = plot_level(
            curves=curves,
            level_kv=level_kv,
            year=args.year,
            unit=args.unit,
            curve_points=args.curve_points,
            output_dir=output_dir,
        )
        created.append(output)
        n_points = sum(curve.unique_timestamps for curve in curves)
        print(
            f"{level_kv:>3} kV: {len(curves):>3} transformatorjev, "
            f"{n_points:>9} veljavnih meritev"
        )

    if warnings:
        print(f"Opozorila: {len(warnings)}")
        for warning in warnings:
            print(f"  - {warning}")

    if not any(curves_by_level.values()):
        raise ValueError(f"Za leto {args.year} ni bilo uporabnih meritev.")

    print("\nUstvarjene datoteke:")
    for path in created:
        print(f"  - {path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"NAPAKA: {error}", file=sys.stderr)
        raise SystemExit(1) from error
