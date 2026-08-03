import argparse
from pathlib import Path

import pandas as pd


# The settings below intentionally match Reactive_Power_Analysis.py.
QUALITY_FILTER = True
INCLUDE_MULTI_VOLTAGE_TRANSFORMERS = False
MV_VOLTAGE_KV_CANDIDATES = [10, 20, 35]
HV_VOLTAGE_KV = 110
VALUES_ARE_IN_KILO = False

# A timestamp is eligible for the system peak only when at least this share of
# the selected 110/MV transformer measurements is available. This prevents a
# data outage from creating a false peak. Use --min-coverage 0 to disable it.
DEFAULT_MIN_COVERAGE = 0.95
DEFAULT_TOP_COUNT = 20


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Find the most negative total reactive-power flow of all 110/MV "
            "transformers and print the RTP contributions at that timestamp."
        )
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing the transformer Parquet component files.",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=DEFAULT_MIN_COVERAGE,
        help=(
            "Minimum required share of available transformer Q measurements "
            f"at the peak timestamp (default: {DEFAULT_MIN_COVERAGE:.2f})."
        ),
    )
    parser.add_argument(
        "--top",
        type=int,
        default=DEFAULT_TOP_COUNT,
        help=f"Number of RTPs to print (default: {DEFAULT_TOP_COUNT}).",
    )
    args = parser.parse_args()
    args.input_dir = args.input_dir.expanduser().resolve()

    if not args.input_dir.is_dir():
        parser.error(f"Input directory does not exist: {args.input_dir}")
    if not 0 <= args.min_coverage <= 1:
        parser.error("--min-coverage must be between 0 and 1.")
    if args.top < 1:
        parser.error("--top must be at least 1.")

    return args


def find_col(df: pd.DataFrame, candidates):
    cols_lower = {str(col).lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in cols_lower:
            return cols_lower[candidate.lower()]
    return None


def parse_meta_from_filename(path: Path):
    """Parse transformer metadata exactly as Reactive_Power_Analysis.py does."""
    parts = path.stem.split("_")
    if len(parts) < 4 or parts[0] != "TR":
        return None

    voltage_idx = [i for i, part in enumerate(parts) if part.isdigit()]
    if not voltage_idx:
        return None

    voltages = [int(parts[i]) for i in voltage_idx]
    if not INCLUDE_MULTI_VOLTAGE_TRANSFORMERS and len(set(voltages)) > 1:
        return None

    i_v = voltage_idx[0]
    rtp = "_".join(parts[1:i_v])
    transformer_id = "_".join(parts[i_v + 1 :])
    if not rtp or not transformer_id:
        return None

    return {
        "component_file_id": path.stem,
        "rtp": rtp,
        "transformer_id": transformer_id,
        "voltage_kv": voltages[0],
    }


def build_110_mv_candidates(all_files):
    """Return strict 110/MV units with no winding above 110 kV."""
    by_key = {}
    for path in all_files:
        meta = parse_meta_from_filename(path)
        if meta is None:
            continue
        key = (meta["rtp"], meta["transformer_id"])
        by_key.setdefault(key, {}).setdefault(meta["voltage_kv"], []).append(
            (path, meta)
        )

    candidates = []
    excluded_110_count = 0
    for by_voltage in by_key.values():
        hv_records = by_voltage.get(HV_VOLTAGE_KV, [])
        if not hv_records:
            continue

        # Reject HV/HV and multi-winding units such as DIVACA TR211
        # (220/110/10 kV).  Its 10 kV tertiary must not make the 110 kV
        # winding look like a distribution 110/MV transformer.
        has_higher_voltage_side = any(
            voltage_kv > HV_VOLTAGE_KV for voltage_kv in by_voltage
        )
        if has_higher_voltage_side:
            excluded_110_count += len(hv_records)
            continue

        has_mv_pair = any(
            mv_voltage in by_voltage for mv_voltage in MV_VOLTAGE_KV_CANDIDATES
        )
        if not has_mv_pair:
            excluded_110_count += len(hv_records)
            continue

        candidates.extend(sorted(hv_records, key=lambda item: item[0].name))

    candidates.sort(key=lambda item: item[0].name)
    return candidates, excluded_110_count


def read_transformer_q(path: Path):
    """Read and clean time/Q using the rules from Reactive_Power_Analysis.py."""
    df = pd.read_parquet(path)
    time_col = find_col(
        df, ["time", "cas", "systime", "systime(UTC+1)", "period_start"]
    )
    q_col = find_col(df, ["Q", "q", "Q_MVAr", "q_mvar"])
    qst_col = find_col(df, ["qst_no", "qst_no_min", "quality", "status"])

    if time_col is None:
        raise RuntimeError(f"No time column in file: {path.name}")
    if q_col is None:
        raise RuntimeError(f"No Q column in file: {path.name}")

    keep_cols = list(dict.fromkeys([time_col, q_col, qst_col]))
    keep_cols = [col for col in keep_cols if col is not None]
    df = df[keep_cols].copy().rename(columns={time_col: "time", q_col: "Q_MVAr"})

    df["time"] = pd.to_datetime(df["time"], errors="coerce", dayfirst=True)
    if df["Q_MVAr"].dtype == "object":
        df["Q_MVAr"] = df["Q_MVAr"].astype(str).str.replace(",", ".", regex=False)
    df["Q_MVAr"] = pd.to_numeric(df["Q_MVAr"], errors="coerce")
    df = df.dropna(subset=["time", "Q_MVAr"]).copy()

    if QUALITY_FILTER and qst_col is not None:
        df[qst_col] = pd.to_numeric(df[qst_col], errors="coerce")
        df = df[df[qst_col] == 1].copy()

    if VALUES_ARE_IN_KILO:
        df["Q_MVAr"] = df["Q_MVAr"] / 1000.0

    return (
        df.groupby("time", as_index=False)["Q_MVAr"]
        .mean()
        .sort_values("time")
        .reset_index(drop=True)
    )


def build_q_matrix(candidates):
    """Build a time-aligned matrix with one Q column per transformer."""
    series = []
    column_meta = {}

    for path, meta in candidates:
        measurements = read_transformer_q(path)
        if measurements.empty:
            continue

        column = meta["component_file_id"]
        series.append(measurements.set_index("time")["Q_MVAr"].rename(column))
        column_meta[column] = meta

    if not series:
        return pd.DataFrame(), {}

    matrix = pd.concat(series, axis=1, join="outer").sort_index()
    matrix.index.name = "time"
    return matrix, column_meta


def print_peak_report(q_matrix, column_meta, min_coverage, top_count):
    transformer_count = q_matrix.shape[1]
    available_count = q_matrix.notna().sum(axis=1)
    coverage = available_count / transformer_count
    total_q = q_matrix.sum(axis=1, min_count=1)

    valid = total_q.notna() & (coverage >= min_coverage)
    if not valid.any():
        raise RuntimeError(
            "No timestamps meet the requested minimum measurement coverage "
            f"({min_coverage:.1%})."
        )

    peak_time = total_q[valid].idxmin()
    peak_total_q = float(total_q.loc[peak_time])
    peak_values = q_matrix.loc[peak_time].dropna()

    rows = []
    for rtp in sorted({meta["rtp"] for meta in column_meta.values()}):
        columns = [
            column for column, meta in column_meta.items() if meta["rtp"] == rtp
        ]
        values = peak_values.reindex(columns).dropna()
        if values.empty:
            continue
        rows.append(
            {
                "RTP": rtp,
                "Q_MVAr": float(values.sum()),
                "measured_transformers": int(values.size),
                "selected_transformers": len(columns),
            }
        )

    contributions = pd.DataFrame(rows).sort_values(
        ["Q_MVAr", "RTP"], ascending=[True, True]
    )
    capacitive_total = -contributions.loc[
        contributions["Q_MVAr"] < 0, "Q_MVAr"
    ].sum()

    # Share_capacitive uses only negative RTP flows as its denominator and thus
    # answers how much each RTP contributes to gross capacitive production.
    contributions["share_capacitive_pct"] = 0.0
    if capacitive_total > 0:
        negative = contributions["Q_MVAr"] < 0
        contributions.loc[negative, "share_capacitive_pct"] = (
            -contributions.loc[negative, "Q_MVAr"] / capacitive_total * 100.0
        )

    # This net share can exceed 100% in total for negative RTPs when positive
    # RTP flows partly offset the system's capacitive peak.
    if peak_total_q != 0:
        contributions["share_net_peak_pct"] = (
            contributions["Q_MVAr"] / peak_total_q * 100.0
        )
    else:
        contributions["share_net_peak_pct"] = float("nan")

    top = contributions[contributions["Q_MVAr"] < 0].head(top_count).copy()
    top_capacitive_share = top["share_capacitive_pct"].sum()
    top_net_share = top["share_net_peak_pct"].sum()
    measured = int(available_count.loc[peak_time])

    print("\nMOST NEGATIVE TOTAL REACTIVE-POWER FLOW")
    print("=" * 78)
    print(f"Timestamp:                {pd.Timestamp(peak_time)}")
    print(f"Total Q:                  {peak_total_q:.3f} MVAr")
    print(
        f"Measurement coverage:     {measured}/{transformer_count} "
        f"transformers ({coverage.loc[peak_time]:.1%})"
    )
    print(f"RTPs with measurements:   {len(contributions)}")
    print(f"Gross capacitive Q:       {-capacitive_total:.3f} MVAr")

    print(f"\nTOP {len(top)} RTP CONTRIBUTORS")
    print("=" * 78)
    print(
        f"{'No.':>3}  {'RTP':<25} {'Q [MVAr]':>12} "
        f"{'Cap. share':>11} {'Net share':>11} {'TR':>7}"
    )
    print("-" * 78)
    for rank, row in enumerate(top.itertuples(index=False), start=1):
        tr_count = f"{row.measured_transformers}/{row.selected_transformers}"
        print(
            f"{rank:>3}  {row.RTP:<25} {row.Q_MVAr:>12.3f} "
            f"{row.share_capacitive_pct:>10.2f}% "
            f"{row.share_net_peak_pct:>10.2f}% {tr_count:>7}"
        )
    print("-" * 78)
    print(f"Top {len(top)} share of gross capacitive Q: {top_capacitive_share:.2f}%")
    print(f"Top {len(top)} share of net negative peak:  {top_net_share:.2f}%")
    print(
        "\nCap. share = share among RTPs with Q < 0. "
        "Net share = Q_RTP / total system Q."
    )


def main():
    args = parse_arguments()
    all_files = sorted(args.input_dir.glob("TR_*.parquet"))
    if not all_files:
        raise FileNotFoundError(
            f"No transformer Parquet files found in: {args.input_dir}"
        )

    candidates, excluded_count = build_110_mv_candidates(all_files)
    if not candidates:
        raise RuntimeError("No matching 110/MV transformer pairs were found.")

    print(f"Selected 110/MV transformer files: {len(candidates)}")
    print(f"Excluded 110 kV files that are not strict 110/MV units: {excluded_count}")
    print("Reading transformer measurements ...")

    q_matrix, column_meta = build_q_matrix(candidates)
    if q_matrix.empty:
        raise RuntimeError("No valid reactive-power measurements were read.")

    print_peak_report(
        q_matrix=q_matrix,
        column_meta=column_meta,
        min_coverage=args.min_coverage,
        top_count=args.top,
    )


if __name__ == "__main__":
    main()
