from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import re

import polars as pl


DEFAULT_START = "2025-04-01"
DEFAULT_END = "2025-04-20"
DEFAULT_MAX_GAP = timedelta(hours=2)
VALID_U_MIN_PU = 0.5
VALID_U_MAX_PU = 1.5


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


def canonical_station(value: str) -> str:
    """Poenoti zapise postaj, na primer PRIMSKOVOGIS in PRIMSKOVO."""
    station = re.sub(r"[^A-Z0-9]", "", value.upper())
    if station.endswith("GIS") and len(station) > 3:
        station = station[:-3]
    return station


def parse_transformer_filename(path: Path) -> TransformerFile | None:
    tokens = path.stem.split("_")
    if not tokens or tokens[0].upper() != "TR":
        return None

    voltage_index = next(
        (
            index
            for index, token in enumerate(tokens[1:], start=1)
            if token.isdigit()
        ),
        None,
    )
    if (
        voltage_index is None
        or voltage_index == 1
        or voltage_index == len(tokens) - 1
    ):
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


def known_topology_stations(
    transformers: list[TransformerFile],
) -> set[str]:
    stations = {canonical_station(item.station) for item in transformers}
    stations.update(
        canonical_station(item.station.removesuffix("GIS"))
        for item in transformers
        if item.station.endswith("GIS")
    )
    return stations


def line_endpoint_from_tokens(
    tokens: list[str],
    known_stations: set[str],
) -> str:
    """Iz dela imena za napetostjo izloči RTP in oznako voda."""
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
    if cleaned and (
        cleaned[-1].isdigit()
        or re.fullmatch(r"[LDK]\d+", cleaned[-1])
    ):
        cleaned.pop()
    if not cleaned:
        raise ValueError(
            "V imenu LINE elementa ni mogoče razbrati končnega RTP-ja."
        )
    return canonical_station("_".join(cleaned))


def parse_line_filename(
    path: Path,
    known_stations: set[str],
) -> tuple[str, str, int] | None:
    tokens = path.stem.split("_")
    if not tokens or tokens[0].upper() != "LINE":
        return None

    voltage_index = next(
        (
            index
            for index, token in enumerate(tokens[1:], start=1)
            if token.isdigit()
        ),
        None,
    )
    if (
        voltage_index is None
        or voltage_index == 1
        or voltage_index == len(tokens) - 1
    ):
        return None

    station_from = canonical_station("_".join(tokens[1:voltage_index]))
    station_to = line_endpoint_from_tokens(
        tokens[voltage_index + 1 :],
        known_stations,
    )
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


def parse_period(
    start_text: str,
    end_text: str,
) -> tuple[datetime, datetime]:
    try:
        start = datetime.fromisoformat(start_text)
        end = datetime.fromisoformat(end_text)
    except ValueError as error:
        raise ValueError(
            "--start in --end morata biti veljavna ISO datuma/časa."
        ) from error

    if "T" not in end_text and " " not in end_text:
        end_exclusive = end + timedelta(days=1)
    else:
        end_exclusive = end

    if end_exclusive <= start:
        raise ValueError("Konec obdobja mora biti za začetkom.")
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
            f"{transformer_file.path.name} nima stolpcev: "
            f"{', '.join(sorted(missing))}."
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


def split_continuous_periods(
    aligned: pl.DataFrame,
    maximum_gap: timedelta,
) -> list[pl.DataFrame]:
    """Razdeli skupne meritve na zvezna časovna obdobja."""
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

    with_period = aligned.with_columns(
        pl.Series("period_id", period_ids)
    )
    periods = []
    for period_id in range(1, current_period + 1):
        period = (
            with_period.filter(pl.col("period_id") == period_id)
            .drop("period_id")
            .sort("time")
        )
        if period.height >= 2:
            periods.append(period)
    return periods
