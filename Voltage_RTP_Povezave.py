from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import re
import sys

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import polars as pl


DEFAULT_START = "2025-04-01"
DEFAULT_END = "2025-04-20"
DEFAULT_RTP_400 = "OKROGLO"
DEFAULT_RTP_110 = "PRIMSKOVOGIS"

VALID_U_MIN_PU = 0.5
VALID_U_MAX_PU = 1.5
DEFAULT_MAX_GAP = timedelta(hours=2)

COLORS = {
    "400": "#2463A8",
    "110_root": "#D97706",
    "110_connected": "#9A7B17",
    "grid": "#D7DCE2",
    "text": "#20252B",
    "reference": "#6B7280",
}


@dataclass(frozen=True)
class TransformerFile:
    path: Path
    station: str
    voltage_kv: int
    transformer: str


@dataclass(frozen=True)
class SeriesSelection:
    transformer_file: TransformerFile
    valid_points: int


def default_component_dir() -> Path:
    return (
        Path(__file__).resolve().parent.parent
        / "Uman meritve"
        / "2026_06_17  SCADA meritve 4600"
        / "urejeno"
        / "Uman_parquet"
        / "component_files"
    )


def default_output_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "Uman meritve" / "grafi_napetosti_RTP"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Izrise napetosti 400/110-kV transformatorja izbranega RTP-ja ter "
            "110-kV transformatorja povezanega RTP-ja. Pot med RTP-jema poisce "
            "iz elementov LINE_*.parquet."
        )
    )
    parser.add_argument(
        "--component-dir",
        type=Path,
        default=default_component_dir(),
        help="Mapa component_files z datotekami TR_*.parquet in LINE_*.parquet.",
    )
    parser.add_argument(
        "--rtp-400",
        default=DEFAULT_RTP_400,
        help="RTP s 400- in 110-kV stranjo (privzeto: OKROGLO).",
    )
    parser.add_argument(
        "--rtp-110",
        default=DEFAULT_RTP_110,
        help="Povezani 110-kV RTP (privzeto: PRIMSKOVOGIS).",
    )
    parser.add_argument(
        "--root-transformer",
        default=None,
        help=(
            "Transformator na osnovnem RTP-ju, npr. TR411. Ce ni podan, se "
            "samodejno izbere objekt z veljavnimi meritvami na 400 in 110 kV."
        ),
    )
    parser.add_argument(
        "--connected-transformer",
        default=None,
        help=(
            "Transformator na povezanem 110-kV RTP-ju, npr. TR2. Ce ni podan, "
            "se izbere tisti z najvec veljavnimi meritvami v obdobju."
        ),
    )
    parser.add_argument(
        "--start",
        default=DEFAULT_START,
        help="Zacetek obdobja v ISO zapisu (privzeto: 2025-04-01).",
    )
    parser.add_argument(
        "--end",
        default=DEFAULT_END,
        help=(
            "Vkljucni konec obdobja v ISO zapisu. Pri samem datumu se uposteva "
            "cel dan (privzeto: 2025-04-20)."
        ),
    )
    parser.add_argument(
        "--max-hops",
        type=int,
        default=4,
        help="Najvecje dovoljeno stevilo 110-kV vodov med RTP-jema (privzeto: 4).",
    )
    parser.add_argument(
        "--max-gap-hours",
        type=float,
        default=DEFAULT_MAX_GAP.total_seconds() / 3600,
        help="Pri vecji casovni luknji se crta prekine (privzeto: 2 h).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir(),
        help="Izhodna mapa za PNG.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Po shranjevanju graf tudi odpre v interaktivnem oknu.",
    )
    return parser.parse_args()


def canonical_station(value: str) -> str:
    """Poenoti zapise postaj, npr. PRIMSKOVOGIS in PRIMSKOVO."""
    station = re.sub(r"[^A-Z0-9]", "", value.upper())
    if station.endswith("GIS") and len(station) > 3:
        station = station[:-3]
    return station


def parse_transformer_filename(path: Path) -> TransformerFile | None:
    tokens = path.stem.split("_")
    if not tokens or tokens[0].upper() != "TR":
        return None

    voltage_index = next(
        (index for index, token in enumerate(tokens[1:], start=1) if token.isdigit()),
        None,
    )
    if voltage_index is None or voltage_index == 1 or voltage_index == len(tokens) - 1:
        return None

    return TransformerFile(
        path=path,
        station="_".join(tokens[1:voltage_index]).upper(),
        voltage_kv=int(tokens[voltage_index]),
        transformer="_".join(tokens[voltage_index + 1 :]).upper(),
    )


def discover_transformers(component_dir: Path) -> list[TransformerFile]:
    result = []
    for path in component_dir.glob("TR_*.parquet"):
        parsed = parse_transformer_filename(path)
        if parsed is not None:
            result.append(parsed)
    return result


def resolve_station(requested: str, transformers: list[TransformerFile]) -> str:
    requested_canonical = canonical_station(requested)
    candidates = sorted(
        {
            item.station
            for item in transformers
            if canonical_station(item.station) == requested_canonical
        }
    )
    if not candidates:
        raise ValueError(
            f"Za RTP {requested!r} ni najdena nobena datoteka TR_*.parquet."
        )

    requested_upper = requested.upper()
    if requested_upper in candidates:
        return requested_upper
    return candidates[0]


def known_topology_stations(transformers: list[TransformerFile]) -> set[str]:
    stations = {canonical_station(item.station) for item in transformers}
    # Nekateri LINE elementi uporabljajo ime brez pripone GIS.
    stations.update(
        canonical_station(item.station.removesuffix("GIS"))
        for item in transformers
        if item.station.endswith("GIS")
    )
    return stations


def line_endpoint_from_tokens(tokens: list[str], known_stations: set[str]) -> str:
    """Iz dela imena za napetostjo izloci RTP in odstrani oznako voda."""
    candidates = []
    compact_endpoint = canonical_station("_".join(tokens))
    for station in known_stations:
        suffix = compact_endpoint.removeprefix(station)
        if compact_endpoint.startswith(station) and (
            not suffix or re.fullmatch(r"(?:\d+|[LDK]\d+)", suffix)
        ):
            candidates.append((len(station), station))

    for end in range(1, len(tokens) + 1):
        candidate = canonical_station("_".join(tokens[:end]))
        if candidate in known_stations:
            candidates.append((len(candidate), candidate))

    if candidates:
        return max(candidates, key=lambda item: item[0])[1]

    cleaned = [token for token in tokens if token]
    if cleaned and (cleaned[-1].isdigit() or re.fullmatch(r"[LDK]\d+", cleaned[-1])):
        cleaned.pop()
    if not cleaned:
        raise ValueError("V imenu LINE elementa ni mogoce razbrati koncnega RTP-ja.")
    return canonical_station("_".join(cleaned))


def parse_line_filename(
    path: Path,
    known_stations: set[str],
) -> tuple[str, str, int] | None:
    tokens = path.stem.split("_")
    if not tokens or tokens[0].upper() != "LINE":
        return None

    voltage_index = next(
        (index for index, token in enumerate(tokens[1:], start=1) if token.isdigit()),
        None,
    )
    if voltage_index is None or voltage_index == 1 or voltage_index == len(tokens) - 1:
        return None

    station_from = canonical_station("_".join(tokens[1:voltage_index]))
    station_to = line_endpoint_from_tokens(tokens[voltage_index + 1 :], known_stations)
    return station_from, station_to, int(tokens[voltage_index])


def build_line_graph(
    component_dir: Path,
    transformers: list[TransformerFile],
    voltage_kv: int,
) -> dict[str, set[str]]:
    known_stations = known_topology_stations(transformers)
    graph: dict[str, set[str]] = {}

    for path in component_dir.glob("LINE_*.parquet"):
        parsed = parse_line_filename(path, known_stations)
        if parsed is None:
            continue
        station_from, station_to, line_voltage_kv = parsed
        if line_voltage_kv != voltage_kv or station_from == station_to:
            continue
        graph.setdefault(station_from, set()).add(station_to)
        graph.setdefault(station_to, set()).add(station_from)

    return graph


def shortest_path(
    graph: dict[str, set[str]],
    start: str,
    destination: str,
    max_hops: int,
) -> list[str]:
    start = canonical_station(start)
    destination = canonical_station(destination)
    queue = deque([[start]])
    visited = {start}

    while queue:
        path = queue.popleft()
        current = path[-1]
        if current == destination:
            return path
        if len(path) - 1 >= max_hops:
            continue

        for neighbour in sorted(graph.get(current, set())):
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(path + [neighbour])

    raise ValueError(
        f"Med {start} in {destination} ni najdene 110-kV poti z najvec "
        f"{max_hops} vodi."
    )


def parse_period(start_text: str, end_text: str) -> tuple[datetime, datetime]:
    try:
        start = datetime.fromisoformat(start_text)
        end = datetime.fromisoformat(end_text)
    except ValueError as error:
        raise ValueError("--start in --end morata biti veljavna ISO datuma/casa.") from error

    # Sam datum pomeni vkljucno celoten koncni dan.
    if "T" not in end_text and " " not in end_text:
        end_exclusive = end + timedelta(days=1)
    else:
        end_exclusive = end

    if end_exclusive <= start:
        raise ValueError("Konec obdobja mora biti za zacetkom.")
    return start, end_exclusive


def valid_voltage_expression(voltage_kv: int) -> pl.Expr:
    return (
        pl.col("U").is_not_null()
        & pl.col("U").is_finite()
        & pl.col("U").is_between(
            voltage_kv * VALID_U_MIN_PU,
            voltage_kv * VALID_U_MAX_PU,
            closed="both",
        )
    )


def valid_point_count(
    transformer_file: TransformerFile,
    start: datetime,
    end_exclusive: datetime,
) -> int:
    schema = pl.scan_parquet(transformer_file.path).collect_schema()
    missing = {"time", "U"} - set(schema.names())
    if missing:
        raise ValueError(
            f"{transformer_file.path.name} nima stolpcev: {', '.join(sorted(missing))}."
        )

    result = (
        pl.scan_parquet(transformer_file.path)
        .filter(
            (pl.col("time") >= start)
            & (pl.col("time") < end_exclusive)
            & valid_voltage_expression(transformer_file.voltage_kv)
        )
        .select(pl.len().alias("n"))
        .collect()
    )
    return int(result.item())


def choose_root_transformer(
    transformers: list[TransformerFile],
    station: str,
    requested_transformer: str | None,
    start: datetime,
    end_exclusive: datetime,
) -> tuple[SeriesSelection, SeriesSelection]:
    station_files = [
        item
        for item in transformers
        if item.station == station and item.voltage_kv in (110, 400)
    ]
    by_key = {
        (item.transformer, item.voltage_kv): item
        for item in station_files
    }
    common_names = sorted(
        {
            item.transformer
            for item in station_files
            if (item.transformer, 110) in by_key and (item.transformer, 400) in by_key
        }
    )

    if requested_transformer is not None:
        requested = requested_transformer.upper()
        if requested not in common_names:
            raise ValueError(
                f"{station} nima para {requested} na 400 in 110 kV. "
                f"Razpolozljivi pari: {', '.join(common_names) or 'ni parov'}."
            )
        common_names = [requested]

    scored = []
    for transformer in common_names:
        high_file = by_key[(transformer, 400)]
        low_file = by_key[(transformer, 110)]
        high_count = valid_point_count(high_file, start, end_exclusive)
        low_count = valid_point_count(low_file, start, end_exclusive)
        scored.append(
            (
                min(high_count, low_count),
                high_count + low_count,
                transformer,
                SeriesSelection(high_file, high_count),
                SeriesSelection(low_file, low_count),
            )
        )

    if not scored or max(item[0] for item in scored) == 0:
        raise ValueError(
            f"Za {station} v izbranem obdobju ni para 400/110-kV napetosti."
        )

    _, _, _, high_selection, low_selection = max(scored)
    return high_selection, low_selection


def choose_connected_transformer(
    transformers: list[TransformerFile],
    station: str,
    requested_transformer: str | None,
    start: datetime,
    end_exclusive: datetime,
) -> SeriesSelection:
    candidates = [
        item
        for item in transformers
        if item.station == station and item.voltage_kv == 110
    ]
    if requested_transformer is not None:
        requested = requested_transformer.upper()
        candidates = [item for item in candidates if item.transformer == requested]
        if not candidates:
            raise ValueError(f"{station} nima 110-kV transformatorja {requested}.")

    scored = [
        SeriesSelection(item, valid_point_count(item, start, end_exclusive))
        for item in candidates
    ]
    if not scored or max(item.valid_points for item in scored) == 0:
        raise ValueError(
            f"Za 110-kV transformatorje RTP {station} v obdobju ni veljavne napetosti."
        )
    return max(
        scored,
        key=lambda item: (
            item.valid_points,
            item.transformer_file.transformer,
        ),
    )


def read_voltage_series(
    selection: SeriesSelection,
    start: datetime,
    end_exclusive: datetime,
) -> pl.DataFrame:
    item = selection.transformer_file
    return (
        pl.scan_parquet(item.path)
        .filter(
            (pl.col("time") >= start)
            & (pl.col("time") < end_exclusive)
            & valid_voltage_expression(item.voltage_kv)
        )
        .group_by("time")
        .agg(pl.col("U").mean())
        .sort("time")
        .collect()
    )


def align_voltage_series(
    root_400_frame: pl.DataFrame,
    root_110_frame: pl.DataFrame,
    connected_110_frame: pl.DataFrame,
) -> pl.DataFrame:
    """Obdrzi samo case, za katere obstajajo vse tri veljavne meritve."""
    return (
        root_400_frame.rename({"U": "U_400"})
        .join(
            root_110_frame.rename({"U": "U_110_root"}),
            on="time",
            how="inner",
        )
        .join(
            connected_110_frame.rename({"U": "U_110_connected"}),
            on="time",
            how="inner",
        )
        .sort("time")
    )


def split_continuous_periods(
    aligned: pl.DataFrame,
    maximum_gap: timedelta,
) -> list[pl.DataFrame]:
    """
    Razdeli skupne meritve na zvezna obdobja.

    Posamezen manjkajoc vzorec ne ustvari nepotrebnega novega grafa. Novo
    obdobje se zacne sele, ko je vrzel vecja od uporabniske meje
    (privzeto 2 uri), zato graf nikoli ne vsebuje vecjega praznega dela.
    """
    if aligned.is_empty():
        return []

    times = aligned.get_column("time").to_list()
    if len(times) < 2:
        return []

    period_ids = [1]
    current_period = 1
    for previous, current in zip(times, times[1:]):
        if current - previous > maximum_gap:
            current_period += 1
        period_ids.append(current_period)

    with_period = aligned.with_columns(pl.Series("period_id", period_ids))
    periods = []
    for period_id in range(1, current_period + 1):
        period = (
            with_period.filter(pl.col("period_id") == period_id)
            .drop("period_id")
            .sort("time")
        )
        # Iz ene same tocke ni mogoce narisati casovnega poteka.
        if period.height >= 2:
            periods.append(period)
    return periods


def focused_voltage_limits(values: list[float], nominal_kv: int) -> tuple[float, float]:
    finite = [value for value in values if value == value]
    lower = min(finite)
    upper = max(finite)
    padding = max((upper - lower) * 0.12, nominal_kv * 0.002)
    return lower - padding, upper + padding


def plot_voltage_panel(
    axis: plt.Axes,
    frame: pl.DataFrame,
    voltage_column: str,
    selection: SeriesSelection,
    color: str,
) -> None:
    item = selection.transformer_file
    times = frame.get_column("time").to_list()
    voltages = frame.get_column(voltage_column).to_list()
    label = f"{item.station} {item.transformer} ({item.voltage_kv} kV)"

    axis.plot(times, voltages, color=color, linewidth=1.25)
    axis.set_title(label, loc="left", fontsize=11, color=COLORS["text"], pad=7)
    axis.set_ylabel("U [kV]")
    axis.set_ylim(*focused_voltage_limits(voltages, item.voltage_kv))
    axis.grid(axis="y", color=COLORS["grid"], linewidth=0.7, alpha=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def output_filename(
    root_station: str,
    connected_station: str,
    start: datetime,
    end: datetime,
    period_index: int,
) -> str:
    return (
        f"napetosti_{canonical_station(root_station)}_"
        f"{canonical_station(connected_station)}_"
        f"{start.date().isoformat()}_{end.date().isoformat()}_"
        f"obdobje_{period_index:02d}.svg"
    )


def make_period_plot(
    period: pl.DataFrame,
    root_400: SeriesSelection,
    root_110: SeriesSelection,
    connected_110: SeriesSelection,
    topology_path: list[str],
    output_path: Path,
    show_plot: bool,
) -> None:
    period_start = period.get_column("time").min()
    period_end = period.get_column("time").max()

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(16, 10.5),
        sharex=True,
        constrained_layout=True,
    )
    figure.patch.set_facecolor("white")
    figure.suptitle(
        "Napetosti izbranega 400/110-kV RTP-ja in povezanega 110-kV RTP-ja",
        fontsize=16,
        color=COLORS["text"],
        weight="bold",
    )
    figure.text(
        0.5,
        0.955,
        (
            f"Skupne meritve: {period_start:%d. %m. %Y %H:%M}–"
            f"{period_end:%d. %m. %Y %H:%M}  |  "
            f"110-kV pot: {' → '.join(topology_path)}"
        ),
        ha="center",
        va="top",
        fontsize=10,
        color=COLORS["reference"],
    )

    plot_voltage_panel(
        axes[0], period, "U_400", root_400, COLORS["400"]
    )
    plot_voltage_panel(
        axes[1], period, "U_110_root", root_110, COLORS["110_root"]
    )
    plot_voltage_panel(
        axes[2],
        period,
        "U_110_connected",
        connected_110,
        COLORS["110_connected"],
    )

    # Enaka skala na obeh 110-kV panelih omogoca neposredno primerjavo.
    root_110_values = period.get_column("U_110_root").to_list()
    connected_110_values = period.get_column("U_110_connected").to_list()
    shared_110_limits = focused_voltage_limits(
        root_110_values + connected_110_values,
        nominal_kv=110,
    )
    axes[1].set_ylim(*shared_110_limits)
    axes[2].set_ylim(*shared_110_limits)

    locator = mdates.AutoDateLocator(minticks=8, maxticks=16)
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    axes[-1].set_xlabel("Cas")
    for axis in axes:
        axis.tick_params(colors=COLORS["text"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, format="svg", bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close(figure)


def make_plots(
    root_400: SeriesSelection,
    root_110: SeriesSelection,
    connected_110: SeriesSelection,
    topology_path: list[str],
    requested_start: datetime,
    requested_end_exclusive: datetime,
    maximum_gap: timedelta,
    output_dir: Path,
    show_plot: bool,
) -> tuple[list[Path], int]:
    aligned = align_voltage_series(
        read_voltage_series(root_400, requested_start, requested_end_exclusive),
        read_voltage_series(root_110, requested_start, requested_end_exclusive),
        read_voltage_series(
            connected_110,
            requested_start,
            requested_end_exclusive,
        ),
    )
    periods = split_continuous_periods(aligned, maximum_gap)
    if not periods:
        raise ValueError(
            "V izbranem obdobju ni zveznega odseka z vsaj dvema skupnima "
            "veljavnima meritvama vseh treh napetosti."
        )

    output_paths = []
    root_station = root_400.transformer_file.station
    connected_station = connected_110.transformer_file.station
    for period_index, period in enumerate(periods, start=1):
        period_start = period.get_column("time").min()
        period_end = period.get_column("time").max()
        output_path = output_dir / output_filename(
            root_station,
            connected_station,
            period_start,
            period_end,
            period_index,
        )
        make_period_plot(
            period,
            root_400,
            root_110,
            connected_110,
            topology_path,
            output_path,
            show_plot,
        )
        output_paths.append(output_path)

    return output_paths, aligned.height


def main() -> None:
    args = parse_args()
    component_dir = args.component_dir.resolve()
    output_dir = args.output_dir.resolve()
    start, end_exclusive = parse_period(args.start, args.end)

    if not component_dir.is_dir():
        raise FileNotFoundError(f"Mapa component_files ne obstaja: {component_dir}")
    if args.max_hops < 1:
        raise ValueError("--max-hops mora biti vsaj 1.")
    if args.max_gap_hours <= 0:
        raise ValueError("--max-gap-hours mora biti vecji od 0.")

    transformers = discover_transformers(component_dir)
    if not transformers:
        raise FileNotFoundError(f"V {component_dir} ni datotek TR_*.parquet.")

    root_station = resolve_station(args.rtp_400, transformers)
    connected_station = resolve_station(args.rtp_110, transformers)
    graph = build_line_graph(component_dir, transformers, voltage_kv=110)
    topology_path = shortest_path(
        graph,
        root_station,
        connected_station,
        max_hops=args.max_hops,
    )

    root_400, root_110 = choose_root_transformer(
        transformers,
        root_station,
        args.root_transformer,
        start,
        end_exclusive,
    )
    connected_110 = choose_connected_transformer(
        transformers,
        connected_station,
        args.connected_transformer,
        start,
        end_exclusive,
    )

    output_paths, common_point_count = make_plots(
        root_400,
        root_110,
        connected_110,
        topology_path,
        start,
        end_exclusive,
        timedelta(hours=args.max_gap_hours),
        output_dir,
        args.show,
    )

    print("=" * 88)
    print("IZRIS NAPETOSTI RTP")
    print(f"Obdobje: {start} <= time < {end_exclusive}")
    print(f"110-kV povezava: {' -> '.join(topology_path)}")
    print(
        f"Osnovni RTP: {root_station} | "
        f"{root_400.transformer_file.transformer} | 400 in 110 kV"
    )
    print(
        f"Povezani RTP: {connected_station} | "
        f"{connected_110.transformer_file.transformer} | 110 kV"
    )
    print(
        "Veljavne tocke: "
        f"400 kV={root_400.valid_points}, "
        f"110 kV={root_110.valid_points}, "
        f"povezani 110 kV={connected_110.valid_points}"
    )
    print(f"Skupne veljavne casovne tocke: {common_point_count}")
    print(f"Stevilo zveznih obdobij / SVG grafov: {len(output_paths)}")
    for output_path in output_paths:
        print(f"Graf: {output_path}")
    print("=" * 88)


if __name__ == "__main__":
    main()
