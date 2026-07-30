"""Global signed-cosphi histograms for 110/MV distribution substations."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HV_VOLTAGE_KV = 110
DEFAULT_MV_LEVELS_KV = [10, 20, 35]
DEFAULT_BIN_WIDTH = 0.01


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Create a global histogram and cumulative histogram of signed "
            "cos(phi) for all 110/MV RTP measurement points."
        )
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing transformer component Parquet files.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help=(
            "Output directory. Defaults to 'global_cosphi_analysis' next to "
            "the input directory."
        ),
    )
    parser.add_argument(
        "--mv-levels",
        type=int,
        nargs="+",
        default=DEFAULT_MV_LEVELS_KV,
        metavar="KV",
        help="MV voltage levels used to confirm a 110/MV pair.",
    )
    parser.add_argument(
        "--bin-width",
        type=float,
        default=DEFAULT_BIN_WIDTH,
        help="Histogram bin width in the interval [-1, 1].",
    )
    parser.add_argument(
        "--values-in-kilo",
        action="store_true",
        help="Convert P from kW to MW and Q from kVAr to MVAr.",
    )
    parser.add_argument(
        "--no-quality-filter",
        action="store_true",
        help="Do not filter quality/status columns to value 1.",
    )

    args = parser.parse_args()
    args.input_dir = args.input_dir.expanduser().resolve()
    if args.output_dir is None:
        args.output_dir = args.input_dir.parent / "global_cosphi_analysis"
    else:
        args.output_dir = args.output_dir.expanduser().resolve()

    if not args.input_dir.is_dir():
        parser.error(f"Input directory does not exist: {args.input_dir}")
    if not 0 < args.bin_width <= 2:
        parser.error("--bin-width must be greater than 0 and at most 2.")

    return args


def find_col(df: pd.DataFrame, candidates):
    columns_lower = {str(column).lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate.lower() in columns_lower:
            return columns_lower[candidate.lower()]
    return None


def parse_transformer_filename(path: Path):
    """Parse TR_<RTP>_<voltage>_<transformer>.parquet filenames."""
    parts = path.stem.split("_")
    if len(parts) < 4 or parts[0] != "TR":
        return None

    voltage_indices = [i for i, part in enumerate(parts) if part.isdigit()]
    if len(voltage_indices) != 1:
        # This also excludes HV/HV files such as TR_X_400_110_TR1.
        return None

    voltage_index = voltage_indices[0]
    rtp = "_".join(parts[1:voltage_index])
    transformer_id = "_".join(parts[voltage_index + 1 :])
    if not rtp or not transformer_id:
        return None

    return {
        "rtp": rtp,
        "transformer_id": transformer_id,
        "voltage_kv": int(parts[voltage_index]),
    }


def find_110_mv_files(all_files, mv_levels):
    """Return 110-kV files that have a matching MV transformer file."""
    files_by_transformer = {}

    for path in all_files:
        metadata = parse_transformer_filename(path)
        if metadata is None:
            continue

        key = (metadata["rtp"], metadata["transformer_id"])
        files_by_transformer.setdefault(key, {}).setdefault(
            metadata["voltage_kv"], []
        ).append(path)

    selected = []
    excluded_without_mv = []

    for (rtp, transformer_id), files_by_voltage in sorted(
        files_by_transformer.items()
    ):
        hv_files = sorted(files_by_voltage.get(HV_VOLTAGE_KV, []))
        if not hv_files:
            continue

        mv_voltage = next(
            (level for level in mv_levels if files_by_voltage.get(level)),
            None,
        )
        if mv_voltage is None:
            excluded_without_mv.extend(hv_files)
            continue

        for hv_path in hv_files:
            selected.append(
                {
                    "path": hv_path,
                    "rtp": rtp,
                    "transformer_id": transformer_id,
                    "mv_voltage_kv": mv_voltage,
                }
            )

    return selected, excluded_without_mv


def read_pq_file(path: Path, quality_filter=True, values_in_kilo=False):
    """Read and normalize the timestamp, P, and Q columns from one file."""
    df = pd.read_parquet(path)
    time_col = find_col(
        df,
        ["time", "cas", "systime", "systime(UTC+1)", "period_start"],
    )
    p_col = find_col(df, ["P", "p", "P_MW", "p_mw"])
    q_col = find_col(df, ["Q", "q", "Q_MVAr", "q_mvar"])
    quality_col = find_col(df, ["qst_no", "qst_no_min", "quality", "status"])

    if time_col is None or p_col is None or q_col is None:
        missing = []
        if time_col is None:
            missing.append("time")
        if p_col is None:
            missing.append("P")
        if q_col is None:
            missing.append("Q")
        raise RuntimeError(f"Missing {', '.join(missing)} column(s) in {path.name}")

    selected_columns = [time_col, p_col, q_col]
    if quality_col is not None:
        selected_columns.append(quality_col)
    selected_columns = list(dict.fromkeys(selected_columns))

    df = df[selected_columns].copy()
    df = df.rename(columns={time_col: "time", p_col: "P_MW", q_col: "Q_MVAr"})
    df["time"] = pd.to_datetime(df["time"], errors="coerce", dayfirst=True)

    for column in ["P_MW", "Q_MVAr"]:
        if df[column].dtype == "object":
            df[column] = df[column].astype(str).str.replace(",", ".", regex=False)
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if quality_filter and quality_col is not None:
        df[quality_col] = pd.to_numeric(df[quality_col], errors="coerce")
        df = df[df[quality_col] == 1].copy()

    df = df.dropna(subset=["time", "P_MW", "Q_MVAr"])
    if values_in_kilo:
        df["P_MW"] /= 1000.0
        df["Q_MVAr"] /= 1000.0

    return (
        df.groupby("time", as_index=False)[["P_MW", "Q_MVAr"]]
        .mean()
        .sort_values("time")
        .reset_index(drop=True)
    )


def build_rtp_timeseries(selected_files, quality_filter, values_in_kilo):
    """Sum all selected 110-kV transformer measurements within each RTP."""
    rtp_data = {}
    errors = []

    for index, record in enumerate(selected_files, start=1):
        path = record["path"]
        rtp = record["rtp"]
        transformer_id = record["transformer_id"]
        print(f"[{index}/{len(selected_files)}] {path.name}")

        try:
            transformer = read_pq_file(
                path,
                quality_filter=quality_filter,
                values_in_kilo=values_in_kilo,
            )
        except Exception as error:
            errors.append((path.name, str(error)))
            print(f"  -> ERROR: {error}")
            continue

        if transformer.empty:
            print("  -> no valid measurements")
            continue

        suffix = f"{transformer_id}__{index}"
        transformer = transformer.rename(
            columns={"P_MW": f"P__{suffix}", "Q_MVAr": f"Q__{suffix}"}
        )

        if rtp not in rtp_data:
            rtp_data[rtp] = transformer
        else:
            rtp_data[rtp] = rtp_data[rtp].merge(
                transformer,
                on="time",
                how="outer",
            )

    return rtp_data, errors


def calculate_global_cosphi(rtp_data):
    """Pool one signed-cosphi observation per valid RTP and timestamp."""
    global_frames = []

    for rtp, df_rtp in sorted(rtp_data.items()):
        p_columns = [column for column in df_rtp if column.startswith("P__")]
        q_columns = [column for column in df_rtp if column.startswith("Q__")]

        valid_p = df_rtp[p_columns].notna().sum(axis=1) > 0
        valid_q = df_rtp[q_columns].notna().sum(axis=1) > 0
        valid = valid_p & valid_q
        if not valid.any():
            continue

        p_sum = df_rtp.loc[valid, p_columns].sum(axis=1, skipna=True)
        q_sum = df_rtp.loc[valid, q_columns].sum(axis=1, skipna=True)
        apparent_power = np.hypot(p_sum, q_sum)
        nonzero = apparent_power > 0

        rtp_result = pd.DataFrame(
            {
                "rtp": rtp,
                "time": df_rtp.loc[valid, "time"],
                "cosphi": p_sum / apparent_power,
            }
        )
        rtp_result = rtp_result.loc[nonzero & rtp_result["cosphi"].notna()].copy()
        rtp_result["cosphi"] = rtp_result["cosphi"].clip(-1.0, 1.0)
        global_frames.append(rtp_result)

    if not global_frames:
        return pd.DataFrame(columns=["rtp", "time", "cosphi"])
    return pd.concat(global_frames, ignore_index=True)


def histogram_bins(bin_width):
    number_of_bins = max(1, int(np.ceil(2.0 / bin_width)))
    return np.linspace(-1.0, 1.0, number_of_bins + 1)


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


def plot_histograms(global_data, output_dir, bin_width):
    values = global_data["cosphi"].to_numpy()
    bins = histogram_bins(bin_width)

    histogram_path = output_dir / "GLOBAL_01_cosphi_histogram_RTP_110_MV.svg"
    cumulative_path = (
        output_dir / "GLOBAL_02_cosphi_cumulative_histogram_RTP_110_MV.svg"
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.hist(
        values,
        bins=bins,
        color="#1f77b4",
        edgecolor="white",
        linewidth=0.35,
    )
    ax.axvline(0, color="#555555", linewidth=0.9)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Predznačeni faktor moči cosφ")
    ax.set_ylabel("Število merilnih točk RTP")
    ax.set_title("Histogram cosφ vseh RTP 110/SN")
    ax.grid(True, axis="y", alpha=0.35)
    save_figure(fig, histogram_path)

    weights = np.full(values.size, 100.0 / values.size)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.hist(
        values,
        bins=bins,
        weights=weights,
        cumulative=True,
        histtype="step",
        linewidth=1.8,
        color="#1f77b4",
    )
    ax.axvline(0, color="#555555", linewidth=0.9)
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Predznačeni faktor moči cosφ")
    ax.set_ylabel("Kumulativni delež merilnih točk RTP / %")
    ax.set_title("Kumulativni histogram cosφ vseh RTP 110/SN")
    ax.grid(True, alpha=0.35)
    save_figure(fig, cumulative_path)

    return histogram_path, cumulative_path


def main():
    args = parse_arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_files = sorted(args.input_dir.glob("*.parquet"))
    selected_files, excluded_without_mv = find_110_mv_files(
        all_files,
        args.mv_levels,
    )

    print(f"Input directory: {args.input_dir}")
    print(f"Parquet files found: {len(all_files)}")
    print(f"110/MV transformer files selected: {len(selected_files)}")
    print(f"110 kV files excluded without an MV pair: {len(excluded_without_mv)}")

    if not selected_files:
        raise RuntimeError(
            "No matching 110/MV transformer files were found. Check the input "
            "directory and --mv-levels."
        )

    rtp_data, errors = build_rtp_timeseries(
        selected_files,
        quality_filter=not args.no_quality_filter,
        values_in_kilo=args.values_in_kilo,
    )
    global_data = calculate_global_cosphi(rtp_data)
    if global_data.empty:
        raise RuntimeError("No valid nonzero P/Q measurements were found.")

    histogram_path, cumulative_path = plot_histograms(
        global_data,
        args.output_dir,
        args.bin_width,
    )

    print()
    print("DONE")
    print(f"RTPs included: {global_data['rtp'].nunique()}")
    print(f"Valid RTP measurement points: {len(global_data)}")
    print(f"Minimum cos(phi): {global_data['cosphi'].min():.6f}")
    print(f"Maximum cos(phi): {global_data['cosphi'].max():.6f}")
    print(f"Read errors: {len(errors)}")
    print(f"Histogram: {histogram_path}")
    print(f"Cumulative histogram: {cumulative_path}")


if __name__ == "__main__":
    main()
