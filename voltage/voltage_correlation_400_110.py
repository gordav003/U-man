from __future__ import annotations

"""
Korelacija napetosti med izbranim 400-kV RTP-jem in povezanim 110/SN RTP-jem.

Skripta:
  1. poišče meritve napetosti izbranega 400-kV RTP-ja,
  2. poišče izbrani ali topološko najbližji 110/SN RTP,
  3. časovno poravna obe napetostni časovni vrsti,
  4. pretvori napetosti v p.u.,
  5. izračuna korelacijo absolutnih napetosti,
  6. izračuna korelacijo sprememb napetosti za izbrane časovne intervale,
  7. izvede linearno regresijo dU110 = beta0 + beta1 * dU400,
  8. preveri korelacijo pri časovnih zamikih,
  9. shrani podatke, povzetke in SVG grafe.

POMEMBNO:
  - Skripta uporablja modul voltage_data.py, ki ga uporablja tudi obstoječa
    skripta voltage_400kv_substation.py.
  - Topološki graf je zgrajen iz 110-kV vodov. Če je povezava izvedena prek
    400/220 in 220/110 kV, je priporočljivo ciljni 110/SN RTP določiti ročno
    z --connected-station.

Primer ročnega izbora:
  python -m voltage.voltage_correlation_400_110 \
      --root-station BERICEVO \
      --connected-station DOMZALE \
      --start 2025-04-01 \
      --end 2025-04-20

Primer avtomatskega izbora najbližjega 110/SN RTP-ja:
  python -m voltage.voltage_correlation_400_110 \
      --root-station BERICEVO \
      --max-hops 3
"""

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import re
import sys
from typing import Iterable

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from continuous_segments import label_continuous_segments
except ModuleNotFoundError:  # Support direct execution from this directory.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from continuous_segments import label_continuous_segments
import polars as pl

try:
    from .voltage_data import (
        DEFAULT_END,
        DEFAULT_MAX_GAP,
        DEFAULT_START,
        SeriesSelection,
        TransformerFile,
        build_line_graph,
        canonical_station,
        default_component_dir,
        discover_transformers,
        parse_period,
        read_voltage_series,
        valid_point_count,
    )
except ImportError:  # Support direct execution from this directory.
    from voltage_data import (
        DEFAULT_END,
        DEFAULT_MAX_GAP,
        DEFAULT_START,
        SeriesSelection,
        TransformerFile,
        build_line_graph,
        canonical_station,
        default_component_dir,
        discover_transformers,
        parse_period,
        read_voltage_series,
        valid_point_count,
    )


# ============================================================================
# NASTAVITVE GRAFOV
# ============================================================================

COLORS = {
    "u400": "#2463A8",
    "u110": "#B44772",
    "points": "#687386",
    "fit": "#20252B",
    "zero": "#8D949C",
    "grid": "#D7DCE2",
    "positive": "#2F7D5C",
    "negative": "#B54A4A",
}

MIN_COMMON_POINTS = 20
MIN_REGRESSION_POINTS = 10


# ============================================================================
# PODATKOVNE STRUKTURE
# ============================================================================

@dataclass(frozen=True)
class ConnectedChoice:
    selection: SeriesSelection
    path: tuple[str, ...]
    sn_levels_kv: tuple[int, ...]
    selection_mode: str


@dataclass(frozen=True)
class RegressionResult:
    n_points: int
    pearson_r: float
    spearman_r: float
    slope: float
    intercept: float
    r_squared: float
    x_std: float
    y_std: float


# ============================================================================
# ARGUMENTI
# ============================================================================


def default_output_dir() -> Path:
    return (
        Path(__file__).resolve().parent.parent
        / "Uman meritve"
        / "korelacija_napetosti_400_110"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analiza korelacije napetosti med izbranim 400-kV RTP-jem in "
            "povezanim 110/SN RTP-jem."
        )
    )
    parser.add_argument(
        "--component-dir",
        type=Path,
        default=default_component_dir(),
        help="Mapa component_files z datotekami TR_*.parquet in LINE_*.parquet.",
    )
    parser.add_argument(
        "--root-station",
        required=True,
        help="Ime izhodiščnega 400-kV RTP-ja, npr. BERICEVO.",
    )
    parser.add_argument(
        "--root-transformer",
        default=None,
        help=(
            "Po želji določi 400-kV transformator, npr. TR1. Če ni podan, "
            "se izbere meritev z največ veljavnimi točkami."
        ),
    )
    parser.add_argument(
        "--connected-station",
        default=None,
        help=(
            "Po želji ročno določi povezani 110/SN RTP. Če ni podan, se "
            "izbere najbližji dosegljivi RTP po 110-kV vodih."
        ),
    )
    parser.add_argument(
        "--connected-transformer",
        default=None,
        help=(
            "Po želji določi transformator v povezanem 110/SN RTP-ju. "
            "Če ni podan, se izbere 110/SN transformator z največ meritvami."
        ),
    )
    parser.add_argument(
        "--start",
        default=DEFAULT_START,
        help=f"Začetek obdobja v ISO zapisu. Privzeto: {DEFAULT_START}.",
    )
    parser.add_argument(
        "--end",
        default=DEFAULT_END,
        help=f"Vključni konec obdobja v ISO zapisu. Privzeto: {DEFAULT_END}.",
    )
    parser.add_argument(
        "--max-hops",
        type=int,
        default=3,
        help="Največ 110-kV vodov pri avtomatskem izboru RTP-ja. Privzeto: 3.",
    )
    parser.add_argument(
        "--change-minutes",
        type=int,
        nargs="+",
        default=[15, 60],
        help=(
            "Časovni intervali za spremembe napetosti v minutah. "
            "Privzeto: 15 60."
        ),
    )
    parser.add_argument(
        "--lag-range-minutes",
        type=int,
        default=120,
        help=(
            "Največji analizirani časovni zamik v obe smeri. "
            "Privzeto: 120 minut."
        ),
    )
    parser.add_argument(
        "--lag-change-minutes",
        type=int,
        default=60,
        help=(
            "Interval spremembe napetosti, uporabljen pri analizi zamikov. "
            "Privzeto: 60 minut."
        ),
    )
    parser.add_argument(
        "--max-gap-hours",
        type=float,
        default=DEFAULT_MAX_GAP.total_seconds() / 3600,
        help=(
            "Vrzel, pri kateri se začne nov zvezni segment. "
            "Privzeto je vrednost iz voltage_data.DEFAULT_MAX_GAP."
        ),
    )
    parser.add_argument(
        "--max-abs-change-pu",
        type=float,
        default=None,
        help=(
            "Po želji izloči spremembe z abs(dU) nad podano vrednostjo p.u. "
            "Privzeto se ekstremov ne izloča."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir(),
        help="Izhodna mapa za CSV in SVG rezultate.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Grafe po shranjevanju tudi prikaže.",
    )
    return parser.parse_args()


# ============================================================================
# POMOŽNE FUNKCIJE ZA IZBOR RTP-JEV IN MERITEV
# ============================================================================


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Z0-9-]+", "_", str(text).upper()).strip("_")


def canonical_transformer(text: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(text).upper())


def transformer_matches(item: TransformerFile, requested: str | None) -> bool:
    if requested is None:
        return True
    return canonical_transformer(item.transformer) == canonical_transformer(requested)


def build_110_sn_candidates(
    transformers: list[TransformerFile],
    start: datetime,
    end_exclusive: datetime,
) -> dict[str, list[tuple[SeriesSelection, tuple[int, ...]]]]:
    """
    Za vsak RTP vrne vse veljavne 110/SN transformatorske meritve.

    Transformator je obravnavan kot 110/SN, če za isti RTP in isti objekt
    obstaja meritev na 110 kV in vsaj ena meritev na nivoju pod 110 kV.
    """
    levels_by_transformer: dict[tuple[str, str], set[int]] = {}
    for item in transformers:
        levels_by_transformer.setdefault(
            (canonical_station(item.station), canonical_transformer(item.transformer)),
            set(),
        ).add(item.voltage_kv)

    result: dict[str, list[tuple[SeriesSelection, tuple[int, ...]]]] = {}

    for item in transformers:
        if item.voltage_kv != 110:
            continue

        key = (
            canonical_station(item.station),
            canonical_transformer(item.transformer),
        )
        levels = levels_by_transformer.get(key, set())
        sn_levels = tuple(sorted(level for level in levels if 0 < level < 110))
        if not sn_levels:
            continue

        count = valid_point_count(item, start, end_exclusive)
        if count <= 0:
            continue

        station_key = canonical_station(item.station)
        result.setdefault(station_key, []).append(
            (SeriesSelection(item, count), sn_levels)
        )

    for station_key in result:
        result[station_key].sort(
            key=lambda record: (
                -record[0].valid_points,
                canonical_transformer(record[0].transformer_file.transformer),
            )
        )

    return result


def choose_root_400(
    transformers: list[TransformerFile],
    station: str,
    requested_transformer: str | None,
    start: datetime,
    end_exclusive: datetime,
) -> SeriesSelection:
    station_key = canonical_station(station)
    candidates: list[SeriesSelection] = []

    for item in transformers:
        if item.voltage_kv != 400:
            continue
        if canonical_station(item.station) != station_key:
            continue
        if not transformer_matches(item, requested_transformer):
            continue

        count = valid_point_count(item, start, end_exclusive)
        if count > 0:
            candidates.append(SeriesSelection(item, count))

    if not candidates:
        transformer_text = (
            f" in transformator {requested_transformer}"
            if requested_transformer
            else ""
        )
        raise ValueError(
            f"Ni veljavne 400-kV meritve za RTP {station}{transformer_text}."
        )

    candidates.sort(
        key=lambda selection: (
            -selection.valid_points,
            canonical_transformer(selection.transformer_file.transformer),
        )
    )
    return candidates[0]


def shortest_path(
    graph: dict[str, set[str]],
    start_station: str,
    target_station: str,
    max_hops: int | None = None,
) -> tuple[str, ...] | None:
    start = canonical_station(start_station)
    target = canonical_station(target_station)

    if start == target:
        return (start,)

    queue: deque[tuple[str, ...]] = deque([(start,)])
    visited = {start}

    while queue:
        path = queue.popleft()
        current = path[-1]
        hops = len(path) - 1

        if max_hops is not None and hops >= max_hops:
            continue

        for neighbour in sorted(graph.get(current, set())):
            if neighbour in visited:
                continue
            next_path = path + (neighbour,)
            if neighbour == target:
                return next_path
            visited.add(neighbour)
            queue.append(next_path)

    return None


def choose_connected_manual(
    station: str,
    requested_transformer: str | None,
    candidates_by_station: dict[
        str,
        list[tuple[SeriesSelection, tuple[int, ...]]],
    ],
    graph: dict[str, set[str]],
    root_station: str,
) -> ConnectedChoice:
    station_key = canonical_station(station)
    candidates = candidates_by_station.get(station_key, [])

    if requested_transformer is not None:
        candidates = [
            record
            for record in candidates
            if transformer_matches(
                record[0].transformer_file,
                requested_transformer,
            )
        ]

    if not candidates:
        transformer_text = (
            f" in transformator {requested_transformer}"
            if requested_transformer
            else ""
        )
        raise ValueError(
            f"Ni veljavne 110/SN meritve za RTP {station}{transformer_text}."
        )

    selection, sn_levels = candidates[0]
    path = shortest_path(graph, root_station, station)

    if path is None:
        # Ročni izbor je dovoljen tudi, kadar poenostavljeni 110-kV graf ne zna
        # dokazati poti, npr. pri povezavi prek 400/220 in 220/110 kV.
        path = (
            canonical_station(root_station),
            canonical_station(station),
        )
        mode = "manual_no_110kv_path"
    else:
        mode = "manual_verified_110kv_path"

    return ConnectedChoice(
        selection=selection,
        path=path,
        sn_levels_kv=sn_levels,
        selection_mode=mode,
    )


def choose_connected_automatic(
    root_station: str,
    graph: dict[str, set[str]],
    candidates_by_station: dict[
        str,
        list[tuple[SeriesSelection, tuple[int, ...]]],
    ],
    max_hops: int,
) -> ConnectedChoice:
    start = canonical_station(root_station)
    queue: deque[tuple[str, ...]] = deque([(start,)])
    visited = {start}

    ranked_candidates: list[
        tuple[
            int,
            int,
            str,
            str,
            tuple[str, ...],
            SeriesSelection,
            tuple[int, ...],
        ]
    ] = []

    while queue:
        path = queue.popleft()
        current = path[-1]
        hops = len(path) - 1

        if hops > 0 and current in candidates_by_station:
            for selection, sn_levels in candidates_by_station[current]:
                item = selection.transformer_file
                ranked_candidates.append(
                    (
                        hops,
                        -selection.valid_points,
                        current,
                        canonical_transformer(item.transformer),
                        path,
                        selection,
                        sn_levels,
                    )
                )

        if hops >= max_hops:
            continue

        for neighbour in sorted(graph.get(current, set())):
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(path + (neighbour,))

    if not ranked_candidates:
        raise ValueError(
            "Ni dosegljivega 110/SN RTP-ja z veljavno 110-kV meritvijo "
            f"v največ {max_hops} korakih od RTP {root_station}. "
            "Uporabi --connected-station za ročni izbor."
        )

    (
        _,
        _,
        _,
        _,
        path,
        selection,
        sn_levels,
    ) = min(
        ranked_candidates,
        key=lambda record: record[:4],
    )

    return ConnectedChoice(
        selection=selection,
        path=path,
        sn_levels_kv=sn_levels,
        selection_mode="automatic_nearest_110kv_path",
    )


# ============================================================================
# BRANJE IN ČASOVNO PORAVNAVANJE
# ============================================================================


def read_named_voltage_series(
    selection: SeriesSelection,
    output_column: str,
    start: datetime,
    end_exclusive: datetime,
) -> pl.DataFrame:
    frame = read_voltage_series(selection, start, end_exclusive)
    return (
        frame.select(
            pl.col("time"),
            pl.col("U").cast(pl.Float64, strict=False).alias(output_column),
        )
        .drop_nulls(["time", output_column])
        .group_by("time")
        .agg(pl.col(output_column).mean())
        .sort("time")
    )


def align_voltage_pair(
    root_400: SeriesSelection,
    connected_110: SeriesSelection,
    start: datetime,
    end_exclusive: datetime,
) -> pd.DataFrame:
    frame_400 = read_named_voltage_series(
        root_400,
        "U400_kV",
        start,
        end_exclusive,
    )
    frame_110 = read_named_voltage_series(
        connected_110,
        "U110_kV",
        start,
        end_exclusive,
    )

    aligned = (
        frame_400.join(frame_110, on="time", how="inner")
        .drop_nulls(["U400_kV", "U110_kV"])
        .sort("time")
    )

    if aligned.height < MIN_COMMON_POINTS:
        raise ValueError(
            "Premalo skupnih meritev 400 in 110 kV: "
            f"{aligned.height}. Zahtevanih je vsaj {MIN_COMMON_POINTS}."
        )

    df = aligned.to_pandas()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["U400_kV"] = pd.to_numeric(df["U400_kV"], errors="coerce")
    df["U110_kV"] = pd.to_numeric(df["U110_kV"], errors="coerce")
    df = (
        df.dropna(subset=["time", "U400_kV", "U110_kV"])
        .sort_values("time")
        .drop_duplicates(subset=["time"], keep="last")
        .reset_index(drop=True)
    )

    df["U400_pu"] = df["U400_kV"] / 400.0
    df["U110_pu"] = df["U110_kV"] / 110.0
    return df


def infer_sample_interval(df: pd.DataFrame) -> pd.Timedelta:
    differences = df["time"].sort_values().diff().dropna()
    differences = differences[differences > pd.Timedelta(0)]

    if differences.empty:
        raise ValueError("Časovnega koraka meritev ni mogoče določiti.")

    interval = differences.median()
    if interval <= pd.Timedelta(0):
        raise ValueError("Neveljaven ocenjeni časovni korak meritev.")

    return interval


def add_continuous_segments(
    df: pd.DataFrame,
    maximum_gap: pd.Timedelta,
) -> pd.DataFrame:
    result = label_continuous_segments(df, maximum_gap=maximum_gap)
    result["time_gap"] = result["time"].diff()
    return result


def add_change_columns(
    df: pd.DataFrame,
    change_minutes: int,
    sample_interval: pd.Timedelta,
) -> tuple[pd.DataFrame, int, pd.Timedelta]:
    """
    Doda dU400 in dU110 za približno podani časovni interval.

    Razlike se računajo samo znotraj istega zveznega segmenta. Poleg tega se
    preveri, da dejanski čas med točkama ustreza ciljnemu intervalu.
    """
    if change_minutes <= 0:
        raise ValueError("Vrednosti --change-minutes morajo biti večje od 0.")

    sample_minutes = sample_interval.total_seconds() / 60.0
    steps = max(1, int(round(change_minutes / sample_minutes)))
    target_delta = sample_interval * steps

    label = f"{change_minutes}min"
    result = df.copy()

    grouped = result.groupby("segment", sort=False, group_keys=False)
    previous_time = grouped["time"].shift(steps)
    previous_u400 = grouped["U400_pu"].shift(steps)
    previous_u110 = grouped["U110_pu"].shift(steps)

    elapsed = result["time"] - previous_time

    # Dovolimo približno četrtino osnovnega koraka odstopanja, najmanj 30 s.
    tolerance = max(
        sample_interval * 0.25,
        pd.Timedelta(seconds=30),
    )
    valid_elapsed = (elapsed - target_delta).abs() <= tolerance

    result[f"dU400_{label}_pu"] = (result["U400_pu"] - previous_u400).where(
        valid_elapsed
    )
    result[f"dU110_{label}_pu"] = (result["U110_pu"] - previous_u110).where(
        valid_elapsed
    )
    result[f"elapsed_{label}_minutes"] = (
        elapsed.dt.total_seconds() / 60.0
    ).where(valid_elapsed)

    return result, steps, target_delta


# ============================================================================
# STATISTIKA
# ============================================================================


def spearman_correlation(x: pd.Series, y: pd.Series) -> float:
    return float(x.rank(method="average").corr(y.rank(method="average")))


def regression_statistics(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    max_abs_change_pu: float | None = None,
) -> tuple[RegressionResult, pd.DataFrame]:
    subset = data[[x_column, y_column]].replace([np.inf, -np.inf], np.nan).dropna()

    if max_abs_change_pu is not None:
        subset = subset[
            (subset[x_column].abs() <= max_abs_change_pu)
            & (subset[y_column].abs() <= max_abs_change_pu)
        ]

    if len(subset) < MIN_REGRESSION_POINTS:
        raise ValueError(
            f"Premalo veljavnih točk za regresijo {x_column} -> {y_column}: "
            f"{len(subset)}."
        )

    x = subset[x_column].to_numpy(dtype=float)
    y = subset[y_column].to_numpy(dtype=float)

    x_std = float(np.std(x, ddof=1))
    y_std = float(np.std(y, ddof=1))

    if not np.isfinite(x_std) or x_std == 0:
        raise ValueError(f"Spremenljivka {x_column} nima variance.")
    if not np.isfinite(y_std) or y_std == 0:
        raise ValueError(f"Spremenljivka {y_column} nima variance.")

    pearson_r = float(np.corrcoef(x, y)[0, 1])
    spearman_r = spearman_correlation(subset[x_column], subset[y_column])
    slope, intercept = np.polyfit(x, y, deg=1)

    predicted = intercept + slope * x
    residual_sum = float(np.sum((y - predicted) ** 2))
    total_sum = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = float(1.0 - residual_sum / total_sum) if total_sum > 0 else np.nan

    result = RegressionResult(
        n_points=len(subset),
        pearson_r=pearson_r,
        spearman_r=spearman_r,
        slope=float(slope),
        intercept=float(intercept),
        r_squared=r_squared,
        x_std=x_std,
        y_std=y_std,
    )
    return result, subset


def calculate_lag_correlations(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    sample_interval: pd.Timedelta,
    lag_range_minutes: int,
) -> pd.DataFrame:
    if lag_range_minutes < 0:
        raise ValueError("--lag-range-minutes ne sme biti negativen.")

    sample_minutes = sample_interval.total_seconds() / 60.0
    max_lag_steps = int(round(lag_range_minutes / sample_minutes))

    rows: list[dict[str, float | int]] = []

    for lag_steps in range(-max_lag_steps, max_lag_steps + 1):
        # Pozitiven lag: primerjamo x(t) z y(t + lag).
        shifted_y = df.groupby("segment", sort=False)[y_column].shift(-lag_steps)
        pair = pd.DataFrame(
            {
                "x": df[x_column],
                "y": shifted_y,
            }
        ).dropna()

        if len(pair) < MIN_REGRESSION_POINTS:
            correlation = np.nan
        elif pair["x"].std(ddof=1) == 0 or pair["y"].std(ddof=1) == 0:
            correlation = np.nan
        else:
            correlation = float(pair["x"].corr(pair["y"]))

        rows.append(
            {
                "lag_steps": lag_steps,
                "lag_minutes": lag_steps * sample_minutes,
                "n_points": len(pair),
                "pearson_r": correlation,
                "abs_pearson_r": abs(correlation) if np.isfinite(correlation) else np.nan,
            }
        )

    return pd.DataFrame(rows)


# ============================================================================
# GRAFI
# ============================================================================


def apply_axis_style(axis: plt.Axes) -> None:
    axis.grid(
        True,
        which="major",
        axis="both",
        color=COLORS["grid"],
        linewidth=0.7,
        alpha=0.85,
    )
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(labelsize=10)


def save_figure(
    figure: plt.Figure,
    output_path: Path,
    show_plot: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, format="svg", bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close(figure)


def plot_time_series(
    df: pd.DataFrame,
    root_label: str,
    connected_label: str,
    output_path: Path,
    show_plot: bool,
) -> None:
    figure, axis = plt.subplots(figsize=(18, 6.5))

    axis.plot(
        df["time"],
        df["U400_pu"],
        linewidth=1.25,
        color=COLORS["u400"],
        label=f"{root_label} – 400 kV",
    )
    axis.plot(
        df["time"],
        df["U110_pu"],
        linewidth=1.25,
        color=COLORS["u110"],
        label=f"{connected_label} – 110 kV",
    )

    axis.set_xlabel("Time", fontsize=11)
    axis.set_ylabel("U / p.u.", fontsize=11)
    apply_axis_style(axis)

    locator = mdates.AutoDateLocator(minticks=5, maxticks=12)
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    axis.legend(loc="best", frameon=False, fontsize=10)

    figure.tight_layout()
    save_figure(figure, output_path, show_plot)


def plot_scatter_regression(
    subset: pd.DataFrame,
    x_column: str,
    y_column: str,
    x_label: str,
    y_label: str,
    result: RegressionResult,
    output_path: Path,
    show_plot: bool,
    zero_lines: bool,
) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 7.2))

    x = subset[x_column].to_numpy(dtype=float)
    y = subset[y_column].to_numpy(dtype=float)

    axis.scatter(
        x,
        y,
        s=12,
        alpha=0.25,
        color=COLORS["points"],
        edgecolors="none",
    )

    x_line = np.linspace(np.nanmin(x), np.nanmax(x), 200)
    y_line = result.intercept + result.slope * x_line
    axis.plot(
        x_line,
        y_line,
        linewidth=1.8,
        color=COLORS["fit"],
        label="Linear regression",
    )

    if zero_lines:
        axis.axhline(0.0, color=COLORS["zero"], linewidth=0.8, alpha=0.7)
        axis.axvline(0.0, color=COLORS["zero"], linewidth=0.8, alpha=0.7)

    stats_text = (
        f"n = {result.n_points}\n"
        f"Pearson r = {result.pearson_r:.3f}\n"
        f"Spearman rₛ = {result.spearman_r:.3f}\n"
        f"R² = {result.r_squared:.3f}\n"
        f"β₁ = {result.slope:.3f}"
    )
    axis.text(
        0.03,
        0.97,
        stats_text,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": COLORS["grid"],
            "alpha": 0.92,
        },
    )

    axis.set_xlabel(x_label, fontsize=11)
    axis.set_ylabel(y_label, fontsize=11)
    apply_axis_style(axis)
    axis.legend(loc="lower right", frameon=False, fontsize=9)

    figure.tight_layout()
    save_figure(figure, output_path, show_plot)


def plot_lag_correlation(
    lag_df: pd.DataFrame,
    output_path: Path,
    show_plot: bool,
) -> None:
    valid = lag_df.dropna(subset=["pearson_r"]).copy()
    if valid.empty:
        return

    best_index = valid["abs_pearson_r"].idxmax()
    best = valid.loc[best_index]

    figure, axis = plt.subplots(figsize=(10, 6.5))
    axis.plot(
        valid["lag_minutes"],
        valid["pearson_r"],
        linewidth=1.5,
        marker="o",
        markersize=3.5,
        color=COLORS["u400"],
    )
    axis.axhline(0.0, color=COLORS["zero"], linewidth=0.8)
    axis.axvline(0.0, color=COLORS["zero"], linewidth=0.8)
    axis.scatter(
        [best["lag_minutes"]],
        [best["pearson_r"]],
        s=55,
        color=COLORS["negative"],
        zorder=5,
        label=(
            f"Maximum |r|: {best['pearson_r']:.3f} at "
            f"{best['lag_minutes']:.0f} min"
        ),
    )

    axis.set_xlabel(
        "Lag of the 110 kV change relative to the 400 kV change / min",
        fontsize=11,
    )
    axis.set_ylabel("Pearson correlation coefficient r", fontsize=11)
    apply_axis_style(axis)
    axis.legend(loc="best", frameon=False, fontsize=9)

    figure.tight_layout()
    save_figure(figure, output_path, show_plot)


# ============================================================================
# POVZETEK IN IZPIS
# ============================================================================


def result_to_row(
    analysis: str,
    x_column: str,
    y_column: str,
    result: RegressionResult,
) -> dict[str, str | int | float]:
    return {
        "analysis": analysis,
        "x_column": x_column,
        "y_column": y_column,
        "n_points": result.n_points,
        "pearson_r": result.pearson_r,
        "spearman_r": result.spearman_r,
        "r_squared": result.r_squared,
        "slope_beta1": result.slope,
        "intercept_beta0": result.intercept,
        "x_std": result.x_std,
        "y_std": result.y_std,
    }


def print_result(label: str, result: RegressionResult) -> None:
    print(f"{label}")
    print(f"  Točk:          {result.n_points}")
    print(f"  Pearson r:     {result.pearson_r:.5f}")
    print(f"  Spearman r_s:  {result.spearman_r:.5f}")
    print(f"  R^2:           {result.r_squared:.5f}")
    print(f"  beta_1:        {result.slope:.5f}")
    print(f"  beta_0:        {result.intercept:.7f}")


def format_path(path: Iterable[str]) -> str:
    return " -> ".join(path)


# ============================================================================
# GLAVNI PROGRAM
# ============================================================================


def main() -> None:
    args = parse_args()

    component_dir = args.component_dir.resolve()
    output_root = args.output_dir.resolve()
    start, end_exclusive = parse_period(args.start, args.end)

    if not component_dir.is_dir():
        raise FileNotFoundError(
            f"Mapa component_files ne obstaja: {component_dir}"
        )
    if args.max_hops < 1:
        raise ValueError("--max-hops mora biti vsaj 1.")
    if args.max_gap_hours <= 0:
        raise ValueError("--max-gap-hours mora biti večji od 0.")
    if not args.change_minutes:
        raise ValueError("Podan mora biti vsaj en interval --change-minutes.")
    if args.lag_change_minutes <= 0:
        raise ValueError("--lag-change-minutes mora biti večji od 0.")
    if args.max_abs_change_pu is not None and args.max_abs_change_pu <= 0:
        raise ValueError("--max-abs-change-pu mora biti večji od 0.")

    transformers = discover_transformers(component_dir)
    if not transformers:
        raise ValueError("V mapi ni bilo najdenih transformatorskih datotek.")

    root_selection = choose_root_400(
        transformers=transformers,
        station=args.root_station,
        requested_transformer=args.root_transformer,
        start=start,
        end_exclusive=end_exclusive,
    )

    candidates_by_station = build_110_sn_candidates(
        transformers,
        start,
        end_exclusive,
    )
    graph_110 = build_line_graph(
        component_dir,
        transformers,
        voltage_kv=110,
    )

    if args.connected_station:
        connected_choice = choose_connected_manual(
            station=args.connected_station,
            requested_transformer=args.connected_transformer,
            candidates_by_station=candidates_by_station,
            graph=graph_110,
            root_station=root_selection.transformer_file.station,
        )
    else:
        connected_choice = choose_connected_automatic(
            root_station=root_selection.transformer_file.station,
            graph=graph_110,
            candidates_by_station=candidates_by_station,
            max_hops=args.max_hops,
        )

    root_file = root_selection.transformer_file
    connected_file = connected_choice.selection.transformer_file

    pair_name = (
        f"{safe_name(root_file.station)}_{safe_name(root_file.transformer)}__"
        f"{safe_name(connected_file.station)}_{safe_name(connected_file.transformer)}"
    )
    output_dir = output_root / pair_name
    output_dir.mkdir(parents=True, exist_ok=True)

    df = align_voltage_pair(
        root_400=root_selection,
        connected_110=connected_choice.selection,
        start=start,
        end_exclusive=end_exclusive,
    )

    sample_interval = infer_sample_interval(df)
    maximum_gap = pd.Timedelta(hours=args.max_gap_hours)
    df = add_continuous_segments(df, maximum_gap)

    # Dodamo vse zahtevane intervale ter interval za analizo zamikov.
    requested_intervals = sorted(
        set(args.change_minutes + [args.lag_change_minutes])
    )
    interval_metadata: dict[int, dict[str, int | float]] = {}

    for minutes in requested_intervals:
        df, steps, actual_delta = add_change_columns(
            df,
            change_minutes=minutes,
            sample_interval=sample_interval,
        )
        interval_metadata[minutes] = {
            "steps": steps,
            "actual_minutes": actual_delta.total_seconds() / 60.0,
        }

    summary_rows: list[dict[str, str | int | float]] = []

    # ----------------------------------------------------------------------
    # Absolutne napetosti
    # ----------------------------------------------------------------------
    absolute_result, absolute_subset = regression_statistics(
        df,
        x_column="U400_pu",
        y_column="U110_pu",
    )
    summary_rows.append(
        result_to_row(
            "absolute_voltage",
            "U400_pu",
            "U110_pu",
            absolute_result,
        )
    )

    # ----------------------------------------------------------------------
    # Spremembe napetosti
    # ----------------------------------------------------------------------
    change_results: dict[int, tuple[RegressionResult, pd.DataFrame]] = {}

    for minutes in sorted(set(args.change_minutes)):
        label = f"{minutes}min"
        x_column = f"dU400_{label}_pu"
        y_column = f"dU110_{label}_pu"

        result, subset = regression_statistics(
            df,
            x_column=x_column,
            y_column=y_column,
            max_abs_change_pu=args.max_abs_change_pu,
        )
        change_results[minutes] = (result, subset)
        summary_rows.append(
            result_to_row(
                f"voltage_change_{minutes}_minutes",
                x_column,
                y_column,
                result,
            )
        )

    # ----------------------------------------------------------------------
    # Časovni zamiki
    # ----------------------------------------------------------------------
    lag_label = f"{args.lag_change_minutes}min"
    lag_x_column = f"dU400_{lag_label}_pu"
    lag_y_column = f"dU110_{lag_label}_pu"

    lag_df = calculate_lag_correlations(
        df=df,
        x_column=lag_x_column,
        y_column=lag_y_column,
        sample_interval=sample_interval,
        lag_range_minutes=args.lag_range_minutes,
    )

    valid_lags = lag_df.dropna(subset=["pearson_r"])
    if valid_lags.empty:
        best_lag_minutes = np.nan
        best_lag_r = np.nan
    else:
        best_lag = valid_lags.loc[valid_lags["abs_pearson_r"].idxmax()]
        best_lag_minutes = float(best_lag["lag_minutes"])
        best_lag_r = float(best_lag["pearson_r"])

    # ----------------------------------------------------------------------
    # Shranjevanje podatkov
    # ----------------------------------------------------------------------
    metadata = {
        "root_station": root_file.station,
        "root_transformer": root_file.transformer,
        "root_voltage_kv": root_file.voltage_kv,
        "connected_station": connected_file.station,
        "connected_transformer": connected_file.transformer,
        "connected_voltage_kv": connected_file.voltage_kv,
        "connected_sn_levels_kv": "/".join(
            str(level) for level in connected_choice.sn_levels_kv
        ),
        "selection_mode": connected_choice.selection_mode,
        "topology_path": format_path(connected_choice.path),
        "start": start.isoformat(sep=" "),
        "end_exclusive": end_exclusive.isoformat(sep=" "),
        "sample_interval_minutes": sample_interval.total_seconds() / 60.0,
        "n_common_points": len(df),
        "n_segments": int(df["segment"].nunique()),
        "best_lag_minutes": best_lag_minutes,
        "best_lag_pearson_r": best_lag_r,
        "max_abs_change_pu_filter": args.max_abs_change_pu,
    }

    for minutes, info in interval_metadata.items():
        metadata[f"change_{minutes}_steps"] = info["steps"]
        metadata[f"change_{minutes}_actual_minutes"] = info["actual_minutes"]

    df.to_csv(
        output_dir / "01_aligned_voltage_timeseries.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(summary_rows).to_csv(
        output_dir / "02_correlation_regression_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    lag_df.to_csv(
        output_dir / "03_lag_correlations.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame([metadata]).to_csv(
        output_dir / "00_metadata.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ----------------------------------------------------------------------
    # Grafi
    # ----------------------------------------------------------------------
    root_label = f"{root_file.station} {root_file.transformer}"
    connected_label = f"{connected_file.station} {connected_file.transformer}"

    plot_time_series(
        df=df,
        root_label=root_label,
        connected_label=connected_label,
        output_path=output_dir / "01_U400_U110_time_series_pu.svg",
        show_plot=args.show,
    )

    plot_scatter_regression(
        subset=absolute_subset,
        x_column="U400_pu",
        y_column="U110_pu",
        x_label=f"U400 – {root_label} / p.u.",
        y_label=f"U110 – {connected_label} / p.u.",
        result=absolute_result,
        output_path=output_dir / "02_scatter_absolute_U400_U110.svg",
        show_plot=args.show,
        zero_lines=False,
    )

    for plot_index, minutes in enumerate(sorted(change_results), start=3):
        result, subset = change_results[minutes]
        label = f"{minutes}min"
        plot_scatter_regression(
            subset=subset,
            x_column=f"dU400_{label}_pu",
            y_column=f"dU110_{label}_pu",
            x_label=f"ΔU400 ({minutes} min) / p.u.",
            y_label=f"ΔU110 ({minutes} min) / p.u.",
            result=result,
            output_path=(
                output_dir
                / f"{plot_index:02d}_scatter_dU400_dU110_{minutes}min.svg"
            ),
            show_plot=args.show,
            zero_lines=True,
        )

    plot_lag_correlation(
        lag_df=lag_df,
        output_path=output_dir / "10_lag_correlation.svg",
        show_plot=args.show,
    )

    # ----------------------------------------------------------------------
    # Konzolni izpis
    # ----------------------------------------------------------------------
    print("=" * 100)
    print("KORELACIJA NAPETOSTI 400 kV - 110 kV")
    print("=" * 100)
    print(
        f"400-kV meritev:  {root_file.station} {root_file.transformer} "
        f"| veljavnih točk pred združitvijo: {root_selection.valid_points}"
    )
    print(
        f"110/SN meritev:  {connected_file.station} {connected_file.transformer} "
        f"| SN nivoji: {connected_choice.sn_levels_kv}"
    )
    print(f"Način izbora:     {connected_choice.selection_mode}")
    print(f"Topološka pot:    {format_path(connected_choice.path)}")
    print(f"Skupnih točk:     {len(df)}")
    print(
        "Časovni korak:   "
        f"{sample_interval.total_seconds() / 60.0:.3f} min"
    )
    print(f"Zveznih segmentov:{int(df['segment'].nunique()):4d}")
    print("-" * 100)

    print_result("ABSOLUTNE NAPETOSTI: U110(U400)", absolute_result)
    print("-" * 100)

    for minutes in sorted(change_results):
        result, _ = change_results[minutes]
        print_result(
            f"SPREMEMBE NAPETOSTI: dU110(dU400), interval {minutes} min",
            result,
        )
        print("-" * 100)

    if np.isfinite(best_lag_r):
        print(
            "NAJVEČJI |r| PRI ČASOVNEM ZAMIKU: "
            f"r = {best_lag_r:.5f}, lag = {best_lag_minutes:.1f} min"
        )
        if best_lag_minutes > 0:
            print(
                "Pozitiven lag pomeni, da je sprememba U110 statistično "
                "zamaknjena za spremembo U400."
            )
        elif best_lag_minutes < 0:
            print(
                "Negativen lag pomeni, da sprememba U110 v podatkih "
                "statistično nastopi pred spremembo U400."
            )
        else:
            print("Največja korelacija je pri sočasnih spremembah.")
    else:
        print("Korelacije pri časovnih zamikih ni bilo mogoče izračunati.")

    print("-" * 100)
    print(f"Izhodna mapa: {output_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()
