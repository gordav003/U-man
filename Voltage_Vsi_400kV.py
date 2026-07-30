from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
import re
import sys

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import polars as pl

from Voltage_RTP_Povezave import (
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
    split_continuous_periods,
    valid_point_count,
)


ROOT_LEVELS_KV = (400, 220, 110)
MIN_PERIOD_POINTS = 8

COLORS = {
    400: "#2463A8",
    220: "#D97706",
    110: "#8A7613",
    "connected": "#B44772",
    "reference": "#687386",
    "grid": "#D7DCE2",
    "text": "#20252B",
}
@dataclass(frozen=True)
class ConnectedChoice:
    selection: SeriesSelection
    path: list[str]
    sn_levels_kv: tuple[int, ...]


@dataclass(frozen=True)
class PlotResult:
    root: TransformerFile
    connected: TransformerFile | None
    paths: tuple[Path, ...]
    status: str


def default_output_dir() -> Path:
    return (
        Path(__file__).resolve().parent.parent
        / "Uman meritve"
        / "grafi_porocilo_1_trafo_na_400kV_RTP_kV_110_SN"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Za vsak 400-kV RTP izbere en reprezentativen transformator in "
            "izdela vektorski SVG graf napetosti v kV. V locenih panelih "
            "prikaze njegove razpolozljive 400/220/110-kV strani in en "
            "povezan RTP 110/SN."
        )
    )
    parser.add_argument(
        "--component-dir",
        type=Path,
        default=default_component_dir(),
        help="Mapa component_files z datotekami TR_*.parquet in LINE_*.parquet.",
    )
    parser.add_argument(
        "--start",
        default=DEFAULT_START,
        help="Zacetek obdobja v ISO zapisu (privzeto: 2025-04-01).",
    )
    parser.add_argument(
        "--end",
        default=DEFAULT_END,
        help="Vkljucni konec obdobja (privzeto: 2025-04-20).",
    )
    parser.add_argument(
        "--anchor-date",
        default="2025-04-20",
        help=(
            "Izrise samo zvezno obdobje, ki vsebuje ta datum "
            "(privzeto: 2025-04-20)."
        ),
    )
    parser.add_argument(
        "--max-hops",
        type=int,
        default=3,
        help=(
            "Najvec vodov do povezanega 110-kV RTP-ja. Prednost ima "
            "neposredna povezava (privzeto: 3)."
        ),
    )
    parser.add_argument(
        "--max-gap-hours",
        type=float,
        default=DEFAULT_MAX_GAP.total_seconds() / 3600,
        help="Vrzel, pri kateri se zacne nov SVG graf (privzeto: 2 h).",
    )
    parser.add_argument(
        "--stations",
        nargs="+",
        default=None,
        help=(
            "Po zelji omeji izris na navedene 400-kV RTP-je, npr. "
            "--stations OKROGLO BERICEVO. Privzeto se obdelajo vsi."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir(),
        help="Izhodna mapa za vektorske SVG grafe.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Grafe po shranjevanju tudi odpre.",
    )
    return parser.parse_args()


def best_110_sn_by_station(
    transformers: list[TransformerFile],
    start: datetime,
    end_exclusive: datetime,
) -> dict[str, tuple[SeriesSelection, tuple[int, ...]]]:
    """
    Za vsak RTP izbere merjeni 110/SN transformator.

    Pogoj 110/SN pomeni, da ima isti RTP in isti objekt poleg 110-kV datoteke
    tudi vsaj eno transformatorsko datoteko na nivoju pod 110 kV.
    """
    levels_by_transformer: dict[tuple[str, str], set[int]] = {}
    for item in transformers:
        levels_by_transformer.setdefault(
            (item.station, item.transformer),
            set(),
        ).add(item.voltage_kv)

    result: dict[str, tuple[SeriesSelection, tuple[int, ...]]] = {}
    for transformer_file in transformers:
        if transformer_file.voltage_kv != 110:
            continue
        sn_levels = tuple(
            sorted(
                level
                for level in levels_by_transformer[
                    (transformer_file.station, transformer_file.transformer)
                ]
                if 0 < level < 110
            )
        )
        if not sn_levels:
            continue
        count = valid_point_count(transformer_file, start, end_exclusive)
        if count == 0:
            continue
        station = canonical_station(transformer_file.station)
        candidate = SeriesSelection(transformer_file, count)
        current = result.get(station)
        if current is None or (
            candidate.valid_points,
            candidate.transformer_file.transformer,
        ) > (
            current[0].valid_points,
            current[0].transformer_file.transformer,
        ):
            result[station] = (candidate, sn_levels)
    return result


def choose_connected_110(
    root_station: str,
    graph: dict[str, set[str]],
    station_selections: dict[str, tuple[SeriesSelection, tuple[int, ...]]],
    max_hops: int,
) -> ConnectedChoice | None:
    """
    Izbere najblizji merjeni 110-kV RTP.

    Razvrscanje: najmanj vodov, nato najvec veljavnih meritev, nato ime RTP.
    """
    start = canonical_station(root_station)
    queue = deque([[start]])
    visited = {start}
    candidates: list[
        tuple[int, int, str, list[str], SeriesSelection, tuple[int, ...]]
    ] = []

    while queue:
        path = queue.popleft()
        current = path[-1]
        hops = len(path) - 1
        if hops > 0 and current in station_selections:
            selection, sn_levels = station_selections[current]
            candidates.append(
                (
                    hops,
                    -selection.valid_points,
                    current,
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
                queue.append(path + [neighbour])

    if not candidates:
        return None
    _, _, _, path, selection, sn_levels = min(candidates)
    return ConnectedChoice(
        selection=selection,
        path=path,
        sn_levels_kv=sn_levels,
    )


def select_root_levels(
    root_400: TransformerFile,
    transformers_by_key: dict[tuple[str, str, int], TransformerFile],
    start: datetime,
    end_exclusive: datetime,
) -> list[SeriesSelection]:
    selections = []
    for voltage_kv in ROOT_LEVELS_KV:
        transformer_file = transformers_by_key.get(
            (root_400.station, root_400.transformer, voltage_kv)
        )
        if transformer_file is None:
            continue
        count = valid_point_count(transformer_file, start, end_exclusive)
        if count:
            selections.append(SeriesSelection(transformer_file, count))
    return selections


def choose_one_root_per_station(
    roots: list[TransformerFile],
    transformers_by_key: dict[tuple[str, str, int], TransformerFile],
    start: datetime,
    end_exclusive: datetime,
) -> list[TransformerFile]:
    """
    Izbere en reprezentativen 400-kV transformator na RTP.

    Prednostni vrstni red:
      1. veljavna 400-kV napetost,
      2. vec razpolozljivih nizjih strani istega transformatorja,
      3. 220-kV stran pred 110-kV stranjo,
      4. vec veljavnih 400-kV tock,
      5. abecedno prvo ime transformatorja.
    """
    by_station: dict[str, list[TransformerFile]] = {}
    for root in roots:
        by_station.setdefault(root.station, []).append(root)

    selected = []
    for station in sorted(by_station):
        scored = []
        for root in by_station[station]:
            valid_400 = valid_point_count(root, start, end_exclusive)
            valid_lower_levels = []
            for voltage_kv in (220, 110):
                lower_file = transformers_by_key.get(
                    (root.station, root.transformer, voltage_kv)
                )
                if (
                    lower_file is not None
                    and valid_point_count(lower_file, start, end_exclusive) > 0
                ):
                    valid_lower_levels.append(voltage_kv)

            score = (
                int(valid_400 > 0),
                len(valid_lower_levels),
                int(220 in valid_lower_levels),
                int(110 in valid_lower_levels),
                valid_400,
            )
            scored.append((score, root.transformer, root))

        best_score = max(item[0] for item in scored)
        best_candidates = [
            item for item in scored if item[0] == best_score
        ]
        selected.append(min(best_candidates, key=lambda item: item[1])[2])

    return selected


def align_kv_series(
    root_selections: list[SeriesSelection],
    connected: SeriesSelection,
    start: datetime,
    end_exclusive: datetime,
) -> tuple[pl.DataFrame, list[tuple[str, SeriesSelection, bool]]]:
    series_specs: list[tuple[str, SeriesSelection, bool]] = []
    aligned: pl.DataFrame | None = None

    for selection in root_selections:
        level = selection.transformer_file.voltage_kv
        column = f"root_{level}"
        frame = read_voltage_series(selection, start, end_exclusive).select(
            "time",
            pl.col("U").alias(column),
        )
        aligned = frame if aligned is None else aligned.join(frame, on="time", how="inner")
        series_specs.append((column, selection, False))

    connected_column = "connected_110"
    connected_frame = read_voltage_series(connected, start, end_exclusive).select(
        "time",
        pl.col("U").alias(connected_column),
    )
    aligned = (
        connected_frame
        if aligned is None
        else aligned.join(connected_frame, on="time", how="inner")
    )
    series_specs.append((connected_column, connected, True))
    return aligned.sort("time"), series_specs


def focused_kv_limits(values: list[float], nominal_kv: int) -> tuple[float, float]:
    lower = min(values)
    upper = max(values)
    padding = max((upper - lower) * 0.10, nominal_kv * 0.002)
    return lower - padding, upper + padding


def safe_part(text: str) -> str:
    return re.sub(r"[^A-Z0-9-]+", "_", text.upper()).strip("_")


def plot_filename(
    root: TransformerFile,
    connected: TransformerFile,
    period_start: datetime,
    period_end: datetime,
    period_index: int,
) -> str:
    return (
        f"napetosti_kv_{safe_part(root.station)}_{safe_part(root.transformer)}_"
        f"{safe_part(connected.station)}_{safe_part(connected.transformer)}_"
        f"{period_start.date().isoformat()}_{period_end.date().isoformat()}_"
        f"obdobje_{period_index:02d}.svg"
    )


def make_period_plot(
    period: pl.DataFrame,
    root: TransformerFile,
    series_specs: list[tuple[str, SeriesSelection, bool]],
    topology_path: list[str],
    connected_sn_levels_kv: tuple[int, ...],
    output_path: Path,
    show_plot: bool,
) -> None:
    period_start = period.get_column("time").min()
    period_end = period.get_column("time").max()

    panel_count = len(series_specs)
    figure, axes = plt.subplots(
        panel_count,
        1,
        figsize=(18, 3.25 * panel_count),
        sharex=True,
    )
    if panel_count == 1:
        axes = [axes]
    figure.subplots_adjust(
        left=0.075,
        right=0.995,
        top=0.99,
        bottom=0.07,
        hspace=0.12,
    )
    figure.patch.set_facecolor("white")

    for axis, (column, selection, is_connected) in zip(axes, series_specs):
        item = selection.transformer_file
        values = period.get_column(column).to_list()
        if is_connected:
            color = COLORS["connected"]
            sn_text = "/".join(map(str, connected_sn_levels_kv))
            y_label = (
                f"{item.station}\n"
                f"110/{sn_text} kV [kV]"
            )
        else:
            color = COLORS[item.voltage_kv]
            y_label = (
                f"{root.station}\n"
                f"{item.voltage_kv} kV [kV]"
            )

        axis.plot(
            period.get_column("time").to_list(),
            values,
            color=color,
            linewidth=1.25,
        )
        axis.set_ylim(*focused_kv_limits(values, item.voltage_kv))
        axis.set_ylabel(y_label, fontsize=10)
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
        axis.tick_params(colors=COLORS["text"])

    # Vsi 110-kV paneli imajo enako skalo za neposredno primerjavo.
    columns_110 = [
        column
        for column, selection, _ in series_specs
        if selection.transformer_file.voltage_kv == 110
    ]
    if len(columns_110) > 1:
        values_110 = [
            value
            for column in columns_110
            for value in period.get_column(column).to_list()
        ]
        shared_limits = focused_kv_limits(values_110, nominal_kv=110)
        for axis, (_, selection, _) in zip(axes, series_specs):
            if selection.transformer_file.voltage_kv == 110:
                axis.set_ylim(*shared_limits)

    locator = mdates.AutoDateLocator(minticks=5, maxticks=12)
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, format="svg", bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close(figure)


def process_transformer(
    root: TransformerFile,
    transformers_by_key: dict[tuple[str, str, int], TransformerFile],
    connected_choice: ConnectedChoice | None,
    start: datetime,
    end_exclusive: datetime,
    anchor_date: date,
    maximum_gap: timedelta,
    output_dir: Path,
    show_plot: bool,
) -> PlotResult:
    if connected_choice is None:
        return PlotResult(
            root=root,
            connected=None,
            paths=(),
            status="ni dosegljivega RTP 110/SN z veljavno 110-kV meritvijo",
        )

    root_selections = select_root_levels(
        root,
        transformers_by_key,
        start,
        end_exclusive,
    )
    if not root_selections or root_selections[0].transformer_file.voltage_kv != 400:
        return PlotResult(
            root=root,
            connected=connected_choice.selection.transformer_file,
            paths=(),
            status="ni veljavne 400-kV napetosti",
        )

    aligned, series_specs = align_kv_series(
        root_selections,
        connected_choice.selection,
        start,
        end_exclusive,
    )
    periods = [
        period
        for period in split_continuous_periods(aligned, maximum_gap)
        if period.height >= MIN_PERIOD_POINTS
        and period.get_column("time").min().date()
        <= anchor_date
        <= period.get_column("time").max().date()
    ]
    if not periods:
        return PlotResult(
            root=root,
            connected=connected_choice.selection.transformer_file,
            paths=(),
            status=(
                "ni skupnega zveznega obdobja meritev, ki vsebuje "
                f"{anchor_date:%d. %m. %Y}"
            ),
        )

    output_paths = []
    connected_file = connected_choice.selection.transformer_file
    for period_index, period in enumerate(periods, start=1):
        period_start = period.get_column("time").min()
        period_end = period.get_column("time").max()
        output_path = output_dir / plot_filename(
            root,
            connected_file,
            period_start,
            period_end,
            period_index,
        )
        make_period_plot(
            period,
            root,
            series_specs,
            connected_choice.path,
            connected_choice.sn_levels_kv,
            output_path,
            show_plot,
        )
        output_paths.append(output_path)

    return PlotResult(
        root=root,
        connected=connected_file,
        paths=tuple(output_paths),
        status="ok",
    )


def main() -> None:
    args = parse_args()
    component_dir = args.component_dir.resolve()
    output_dir = args.output_dir.resolve()
    start, end_exclusive = parse_period(args.start, args.end)
    try:
        anchor_date = datetime.fromisoformat(args.anchor_date).date()
    except ValueError as error:
        raise ValueError("--anchor-date mora biti veljaven ISO datum.") from error
    if not (start.date() <= anchor_date < end_exclusive.date()):
        raise ValueError("--anchor-date mora biti znotraj obdobja --start/--end.")

    if not component_dir.is_dir():
        raise FileNotFoundError(f"Mapa component_files ne obstaja: {component_dir}")
    if args.max_hops < 1:
        raise ValueError("--max-hops mora biti vsaj 1.")
    if args.max_gap_hours <= 0:
        raise ValueError("--max-gap-hours mora biti vecji od 0.")

    transformers = discover_transformers(component_dir)
    transformers_by_key = {
        (item.station, item.transformer, item.voltage_kv): item
        for item in transformers
    }
    requested_stations = (
        {canonical_station(station) for station in args.stations}
        if args.stations
        else None
    )
    roots = sorted(
        (
            item
            for item in transformers
            if item.voltage_kv == 400
            and (
                requested_stations is None
                or canonical_station(item.station) in requested_stations
            )
        ),
        key=lambda item: (item.station, item.transformer),
    )
    if not roots:
        raise ValueError("Ni najdenih zahtevanih 400-kV transformatorjev.")
    roots = choose_one_root_per_station(
        roots,
        transformers_by_key,
        start,
        end_exclusive,
    )

    graph = build_line_graph(component_dir, transformers, voltage_kv=110)
    station_selections = best_110_sn_by_station(
        transformers,
        start,
        end_exclusive,
    )
    choices_by_station = {
        canonical_station(root.station): choose_connected_110(
            root.station,
            graph,
            station_selections,
            args.max_hops,
        )
        for root in roots
    }

    results = [
        process_transformer(
            root,
            transformers_by_key,
            choices_by_station[canonical_station(root.station)],
            start,
            end_exclusive,
            anchor_date,
            timedelta(hours=args.max_gap_hours),
            output_dir,
            args.show,
        )
        for root in roots
    ]

    created = sum(len(result.paths) for result in results)
    successful = sum(result.status == "ok" for result in results)
    print("=" * 100)
    print("IZRIS ENEGA TRANSFORMATORJA NA VSAK 400-kV RTP V kV")
    print(f"Zahtevano obdobje: {start} <= time < {end_exclusive}")
    print(f"Izbrano je samo zvezno obdobje, ki vsebuje: {anchor_date}")
    print(f"Obdelanih 400-kV RTP-jev: {len(results)}")
    print(f"RTP-jev z izdelanim grafom: {successful}")
    print(f"Ustvarjenih SVG grafov: {created}")
    print(f"Izhodna mapa: {output_dir}")
    print("-" * 100)
    for result in results:
        connected_text = (
            f"{result.connected.station} {result.connected.transformer}"
            if result.connected is not None
            else "-"
        )
        print(
            f"{result.root.station:12s} {result.root.transformer:12s} | "
            f"povezani: {connected_text:24s} | {result.status}"
        )
        for path in result.paths:
            print(f"  {path.name}")
    print("=" * 100)


if __name__ == "__main__":
    main()
