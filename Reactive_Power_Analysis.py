import argparse
from pathlib import Path
import re

# ============================================================
# SETTINGS
# ============================================================

def parse_arguments():
    """Parse input and output directories supplied on the command line."""
    parser = argparse.ArgumentParser(
        description=(
            "Analyze transformer Parquet measurements and reactive-power "
            "deviation for each RTP."
        )
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing the transformer Parquet component files.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help=(
            "Directory for CSV reports and plots. Defaults to an "
            "'rtp_reactive_power_analysis' folder next to the input directory."
        ),
    )
    args = parser.parse_args()
    args.input_dir = args.input_dir.expanduser().resolve()
    if args.output_dir is None:
        args.output_dir = args.input_dir.parent / "rtp_reactive_power_analysis"
    else:
        args.output_dir = args.output_dir.expanduser().resolve()
    return args


ARGS = parse_arguments()
INPUT_DIR = ARGS.input_dir
OUTPUT_DIR = ARGS.output_dir

if not INPUT_DIR.is_dir():
    raise NotADirectoryError(f"Input directory does not exist: {INPUT_DIR}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.ioff()

COSPHI_LIMIT = 0.95

QUALITY_FILTER = True

# Keep this False to exclude files such as TR_X_400_110_TR1 or TR_X_220_110_TR1.
INCLUDE_MULTI_VOLTAGE_TRANSFORMERS = False

# Include only HV/MV transformers in the analysis.
# A 110 kV file is a candidate only when a matching MV file also exists
# for the same RTP and transformer, for example:
#   TR_PRIMSKOVOGIS_110_TR1.parquet + TR_PRIMSKOVOGIS_20_TR1.parquet
ONLY_110_MV_TRANSFORMERS = True

# MV voltage levels that confirm that a 110 kV file belongs to an HV/MV transformer.
# Use [20] for 20 kV only, or include other available levels such as [10, 20, 35].
MV_VOLTAGE_KV_CANDIDATES = [10, 20, 35]

# RTP analysis uses measurements from the 110 kV side.
HV_VOLTAGE_KV = 110

# Parquet values are assumed to already use MW/MVAr.
# Set this to True if P is in kW and Q is in kVAr.
VALUES_ARE_IN_KILO = False

# Using abs(P) is safer for bidirectional flows because the power-factor limit depends on active-power magnitude.
# Keep this False to reproduce the MATLAB calculation without abs(P).
USE_ABS_P_FOR_LIMITS = False

# Minimum number of points required for RTP analysis.
MIN_POINTS = 10

# Minimum number of points required to plot a continuous transformer segment.
MIN_POINTS_SEGMENT = 2

# Start a new segment when the measurement gap exceeds this interval.
MAX_GAP = pd.Timedelta("1h")

# Save RTP and transformer time series as CSV files.
SAVE_RTP_CSV = True
SAVE_TRANSFORMER_CSV = False

# Plot a combined HV/MV comparison for every continuous time segment.
PLOT_HV_MV_COMPARISON = True

# Publication-oriented settings for P(t), Q(t), and U(t) time-series plots.
# Time-series plots have no titles; RTP duration-curve titles remain enabled.
TIME_SERIES_FIGSIZE = (12, 7.5)
TIME_SERIES_PAIR_FIGSIZE = (12, 10.5)
TIME_SERIES_AXIS_LABEL_FONTSIZE = 11
TIME_SERIES_TICK_FONTSIZE = 10
TIME_SERIES_LEGEND_FONTSIZE = 10
TIME_SERIES_LINEWIDTH = 1.2
TIME_SERIES_DATE_INTERVAL_DAYS = 5
VOLTAGE_Y_PADDING_FRACTION = 0.08
VOLTAGE_Y_MIN_PADDING_PU = 0.002

# P/Q sign applied to the MV side in comparison plots.
# Use 1.0 when HV and MV measurements use the same flow orientation.
# Use -1.0 when the MV measurement direction is opposite to the HV direction.
MV_POWER_SIGN_FOR_COMPARISON = 1.0

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_name(s: str) -> str:
    s = str(s)
    s = re.sub(r"[^\w\-\.]+", "_", s, flags=re.UNICODE)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def time_name(ts) -> str:
    if pd.isna(ts):
        return "NA"
    return pd.Timestamp(ts).strftime("%Y-%m-%d_%H-%M")


def configure_shared_time_axis(ax, start_time, end_time):
    """Format a shared time axis with sparse dates and a single year label."""
    start_year = pd.Timestamp(start_time).year
    end_year = pd.Timestamp(end_time).year
    year_label = str(start_year) if start_year == end_year else f"{start_year}\u2013{end_year}"

    ax.xaxis.set_major_locator(
        mdates.DayLocator(interval=TIME_SERIES_DATE_INTERVAL_DAYS)
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_xlabel(
        f"Time / {year_label}",
        fontsize=TIME_SERIES_AXIS_LABEL_FONTSIZE,
    )
    ax.tick_params(axis="x", labelsize=TIME_SERIES_TICK_FONTSIZE)


def configure_voltage_axis(ax, voltage_values):
    """Use a tight data-driven U scale and show only visible reference lines."""
    values = pd.to_numeric(
        pd.Series(np.asarray(voltage_values).ravel()),
        errors="coerce",
    ).dropna()
    if values.empty:
        return

    voltage_min = values.min()
    voltage_max = values.max()
    voltage_span = voltage_max - voltage_min
    padding = max(
        voltage_span * VOLTAGE_Y_PADDING_FRACTION,
        VOLTAGE_Y_MIN_PADDING_PU,
    )
    y_min = voltage_min - padding
    y_max = voltage_max + padding
    ax.set_ylim(y_min, y_max)

    for reference, alpha in ((0.95, 0.35), (1.00, 0.60), (1.05, 0.35)):
        if y_min <= reference <= y_max:
            ax.axhline(
                reference,
                linestyle="--",
                linewidth=1.0,
                color="#2ca02c",
                alpha=alpha,
            )


def save_figure(fig, out_png: Path):
    """Save a preview PNG and an editable vector SVG with the same stem."""
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_png.with_suffix(".svg"), format="svg")
    plt.close(fig)


def find_col(df: pd.DataFrame, candidates):
    cols_lower = {c.lower(): c for c in df.columns}

    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]

    return None


def parse_meta_from_filename(path: Path):
    """
    Read basic metadata from a transformer component filename.

    Supported examples:
      TR_AJDOVSCINA_110_TR1.parquet
      TR_PRIMSKOVOGIS_20_TR2.parquet
      TR_X_400_110_TR1.parquet  -> multi-voltage, excluded by default

    Returns:
      component_file_id, rtp, transformer_id, voltage_kv, n_voltage_tokens, voltages
    """
    stem = path.stem
    parts = stem.split("_")

    if len(parts) < 4:
        return None

    if parts[0] != "TR":
        return None

    voltage_idx = [i for i, part in enumerate(parts) if part.isdigit()]

    if not voltage_idx:
        return None

    voltages = [int(parts[i]) for i in voltage_idx]

    # Exclude files such as 400_110 or 220_110 from this analysis,
    # because they represent HV/HV rather than HV/MV transformers.
    if not INCLUDE_MULTI_VOLTAGE_TRANSFORMERS and len(set(voltages)) > 1:
        return None

    # Standard filenames should contain one voltage token: 110, 20, 10, etc.
    i_v = voltage_idx[0]
    voltage_kv = voltages[0]

    rtp = "_".join(parts[1:i_v])
    transformer_id = "_".join(parts[i_v + 1:])

    if not rtp or not transformer_id:
        return None

    return {
        "component_file_id": stem,
        "rtp": rtp,
        "transformer_id": transformer_id,
        "voltage_kv": voltage_kv,
        "n_voltage_tokens": len(voltage_idx),
        "voltages": voltages,
    }


def build_110_mv_candidates(all_files):
    """
    Find matching measurement pairs for 110/MV transformers.

    Criteria:
      - a transformer file exists at HV_VOLTAGE_KV, for example 110 kV
      - a transformer file exists at one of MV_VOLTAGE_KV_CANDIDATES
      - the matching key is (rtp, transformer_id), for example (PRIMSKOVOGIS, TR1)

    Return a list of triples:
      (hv_path, mv_path, meta)

    The MV file is used not only to confirm the pair; the main program also
    reads, summarizes, and plots its measurements.
    """
    parsed_records = []
    ignored_records = []

    for path in all_files:
        meta = parse_meta_from_filename(path)
        if meta is None:
            ignored_records.append({
                "file": path.name,
                "reason": "not_transformer_or_invalid_format_or_multi_voltage",
            })
            continue

        parsed_records.append({
            "path": path,
            "meta": meta,
            "key": (meta["rtp"], meta["transformer_id"]),
            "voltage_kv": int(meta["voltage_kv"]),
        })

    by_key = {}
    for rec in parsed_records:
        by_key.setdefault(rec["key"], {}).setdefault(rec["voltage_kv"], []).append(rec)

    candidate_pairs = []
    excluded_110 = []
    duplicate_mv_rows = []

    for key, by_voltage in sorted(by_key.items()):
        rtp, transformer_id = key

        hv_recs = by_voltage.get(HV_VOLTAGE_KV, [])
        if not hv_recs:
            continue

        found_mv_voltage = None
        found_mv_recs = []
        for mv_kv in MV_VOLTAGE_KV_CANDIDATES:
            if mv_kv in by_voltage:
                found_mv_voltage = mv_kv
                found_mv_recs = sorted(
                    by_voltage[mv_kv],
                    key=lambda x: x["path"].name,
                )
                break

        if found_mv_voltage is None:
            for rec in hv_recs:
                excluded_110.append({
                    "file": rec["path"].name,
                    "rtp": rtp,
                    "transformer_id": transformer_id,
                    "reason": "no_matching_MV_file",
                    "available_voltages_for_same_rtp_transformer": sorted(by_voltage.keys()),
                })
            continue

        # One MV file is normally expected for each (RTP, transformer, MV level).
        # If several exist, select the first by filename and record the choice in the audit.
        mv_rec = found_mv_recs[0]
        if len(found_mv_recs) > 1:
            duplicate_mv_rows.append({
                "rtp": rtp,
                "transformer_id": transformer_id,
                "mv_voltage_kv": found_mv_voltage,
                "selected_mv_file": mv_rec["path"].name,
                "all_mv_files": ";".join(x["path"].name for x in found_mv_recs),
                "reason": "multiple_MV_files_for_key_selected_first",
            })

        for hv_rec in sorted(hv_recs, key=lambda x: x["path"].name):
            meta = hv_rec["meta"].copy()
            meta["hv_component_file_id"] = hv_rec["meta"]["component_file_id"]
            meta["hv_file"] = hv_rec["path"].name
            meta["mv_voltage_kv"] = found_mv_voltage
            meta["mv_component_file_id"] = mv_rec["meta"]["component_file_id"]
            meta["mv_file"] = mv_rec["path"].name

            candidate_pairs.append((hv_rec["path"], mv_rec["path"], meta))

    stats = {
        "parsed_tr_files": len(parsed_records),
        "ignored_files": len(ignored_records),
        "keys_with_tr_files": len(by_key),
        "candidate_110_mv": len(candidate_pairs),
        "excluded_110_without_mv": len(excluded_110),
        "excluded_110_without_mv_rows": excluded_110,
        "duplicate_mv_rows": duplicate_mv_rows,
        "ignored_rows": ignored_records,
    }

    return candidate_pairs, stats

def convert_voltage_to_kv_and_pu(u_series: pd.Series, un_kv: float):
    """
    Automatically detect the voltage unit:
      - around 1      -> already p.u.
      - around 110    -> kV
      - around 110000 -> V

    Return U_kV, U_pu, and detected_unit.
    """
    u = pd.to_numeric(u_series, errors="coerce")
    u_med = u.dropna().abs().median()

    if pd.isna(u_med):
        return pd.Series(np.nan, index=u.index), pd.Series(np.nan, index=u.index), "missing"

    if 0.5 <= u_med <= 1.5:
        # U is most likely already in p.u.
        u_pu = u
        u_kv = u * un_kv
        detected_unit = "pu"
    elif u_med > 1000:
        # U is most likely in V.
        u_kv = u / 1000.0
        u_pu = u_kv / un_kv
        detected_unit = "V"
    else:
        # U is most likely in kV.
        u_kv = u
        u_pu = u_kv / un_kv
        detected_unit = "kV"

    return u_kv, u_pu, detected_unit


def read_transformer_file(path: Path, un_kv: float):
    """
    Return a DataFrame with:
      time, P_MW, Q_MVAr, U_kV, U_pu

    If no U column exists, U_kV and U_pu remain NaN.
    """
    df = pd.read_parquet(path)

    time_col = find_col(df, ["time", "cas", "systime", "systime(UTC+1)", "period_start"])
    p_col = find_col(df, ["P", "p", "P_MW", "p_mw"])
    q_col = find_col(df, ["Q", "q", "Q_MVAr", "q_mvar"])
    u_col = find_col(df, ["U", "u", "U_kV", "u_kv", "U_pu", "u_pu", "voltage", "Voltage"])

    if time_col is None:
        raise RuntimeError(f"No time column in file: {path.name}")

    if p_col is None:
        raise RuntimeError(f"No P column in file: {path.name}")

    if q_col is None:
        raise RuntimeError(f"No Q column in file: {path.name}")

    keep_cols = [time_col, p_col, q_col]

    if u_col is not None:
        keep_cols.append(u_col)

    qst_col = find_col(df, ["qst_no", "qst_no_min", "quality", "status"])
    if qst_col is not None:
        keep_cols.append(qst_col)

    # Remove duplicate column names if a selected column is repeated.
    keep_cols = list(dict.fromkeys(keep_cols))

    df = df[keep_cols].copy()

    rename_map = {
        time_col: "time",
        p_col: "P_MW",
        q_col: "Q_MVAr",
    }

    if u_col is not None:
        rename_map[u_col] = "U_raw"

    df = df.rename(columns=rename_map)

    df["time"] = pd.to_datetime(df["time"], errors="coerce", dayfirst=True)

    numeric_cols = ["P_MW", "Q_MVAr"]
    if "U_raw" in df.columns:
        numeric_cols.append("U_raw")

    for col in numeric_cols:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # P and Q are required; U is optional.
    df = df.dropna(subset=["time", "P_MW", "Q_MVAr"]).copy()

    if QUALITY_FILTER and qst_col is not None:
        df[qst_col] = pd.to_numeric(df[qst_col], errors="coerce")
        df = df[df[qst_col] == 1].copy()

    if df.empty:
        return pd.DataFrame(
            columns=["time", "P_MW", "Q_MVAr", "U_kV", "U_pu", "U_detected_unit"]
        )

    if VALUES_ARE_IN_KILO:
        df["P_MW"] = df["P_MW"] / 1000.0
        df["Q_MVAr"] = df["Q_MVAr"] / 1000.0

    if "U_raw" in df.columns:
        df["U_kV"], df["U_pu"], detected_unit = convert_voltage_to_kv_and_pu(
            df["U_raw"],
            un_kv=un_kv,
        )
        df["U_detected_unit"] = detected_unit
    else:
        df["U_kV"] = np.nan
        df["U_pu"] = np.nan
        df["U_detected_unit"] = "missing"

    # Average duplicate timestamps.
    df = (
        df.groupby("time", as_index=False)[["P_MW", "Q_MVAr", "U_kV", "U_pu"]]
        .mean()
        .sort_values("time")
        .reset_index(drop=True)
    )

    return df


def add_segments(df: pd.DataFrame, time_col: str = "time"):
    df = df.sort_values(time_col).copy()
    df["dt"] = df[time_col].diff()
    df["segment"] = (df["dt"] > MAX_GAP).cumsum() + 1
    return df


def calculate_dq_from_cosphi_limit(df: pd.DataFrame, cosphi_limit: float):
    """
    Calculate the Q deviation from the cosφ >= cosphi_limit range.

    Input:
      df must contain P_MW_sum and Q_MVAr_sum

    Output:
      adds:
        Q_limit_ind_MVAr
        Q_limit_cap_MVAr
        dQ_MVAr
        status_q
    """
    df = df.copy()

    k = np.sqrt(1 - cosphi_limit**2) / cosphi_limit

    if USE_ABS_P_FOR_LIMITS:
        p_ref = df["P_MW_sum"].abs()
    else:
        p_ref = df["P_MW_sum"]

    q_limit_ind = p_ref * k
    q_limit_cap = -p_ref * k

    q = df["Q_MVAr_sum"]

    dq = pd.Series(0.0, index=df.index)
    status = pd.Series("Within limits", index=df.index)

    mask_cap = q < q_limit_cap
    mask_ind = q > q_limit_ind

    dq.loc[mask_cap] = q.loc[mask_cap] - q_limit_cap.loc[mask_cap]
    dq.loc[mask_ind] = q.loc[mask_ind] - q_limit_ind.loc[mask_ind]

    status.loc[mask_cap] = "Overcompensated (cap)"
    status.loc[mask_ind] = "Undercompensated (ind)"

    df["Q_limit_ind_MVAr"] = q_limit_ind
    df["Q_limit_cap_MVAr"] = q_limit_cap
    df["dQ_MVAr"] = dq
    df["status_q"] = status

    return df


def duration_curve_data(df: pd.DataFrame):
    """
    Sort dQ in ascending order and prepare the x-axis as a percentage of time.
    """
    s = df["dQ_MVAr"].sort_values().reset_index(drop=True)
    n = len(s)

    if n <= 1:
        time_percent = pd.Series([0.0] * n)
    else:
        time_percent = pd.Series(np.arange(n) / (n - 1) * 100.0)

    return time_percent, s


def plot_duration_curve_together(rtp: str, df: pd.DataFrame, out_png: Path):
    """
    Plot the dQ duration curve for an entire RTP.
    """
    time_percent, dq_sorted = duration_curve_data(df)

    under_limit = dq_sorted > 0
    over_limit = dq_sorted < 0
    within_limits = dq_sorted == 0

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(
        time_percent[under_limit],
        dq_sorted[under_limit],
        linewidth=1.2,
        label="Undercompensated (ind)",
        color="#1f77b4",
    )

    ax.plot(
        time_percent[within_limits],
        dq_sorted[within_limits],
        linewidth=1.2,
        label="Within limits",
        color="#ff7f0e",
    )

    ax.plot(
        time_percent[over_limit],
        dq_sorted[over_limit],
        linewidth=1.2,
        label="Overcompensated (cap)",
        color="#f2c14e",
    )

    if over_limit.any():
        ax.axvline(time_percent[over_limit].max(), color="red", alpha=0.55, linewidth=1.2)

    if under_limit.any():
        ax.axvline(time_percent[under_limit].min(), color="red", alpha=0.55, linewidth=1.2)

    ax.grid(True, alpha=0.35)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Percentage of measurement period / %")
    ax.set_ylabel("ΔQ / MVAr")
    ax.set_title(f"Excess reactive-power consumption/generation - RTP {rtp}")
    ax.legend(loc="upper left")

    save_figure(fig, out_png)


def plot_q_status_share(rtp: str, pct_cap: float, pct_ok: float, pct_ind: float, out_png: Path):
    """
    Plot the share of time in each Q status for an entire RTP.
    """
    labels = ["Overcompensated\ncap", "Within limits", "Undercompensated\nind"]
    values = [pct_cap, pct_ok, pct_ind]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, values)

    ax.set_ylim(0, max(100, max(values) * 1.15 if values else 100))
    ax.set_ylabel("Share of time / %")
    ax.set_title(f"Reactive-power state shares - RTP {rtp}")
    ax.grid(True, axis="y", alpha=0.35)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    save_figure(fig, out_png)


def plot_transformer_segment_pq(
    rtp: str,
    comp_id: str,
    segment_id: int,
    df_seg: pd.DataFrame,
    out_png: Path,
    side_label: str,
):
    """Plot P above Q on aligned axes for the selected transformer side."""
    start_time = df_seg["time"].min()
    end_time = df_seg["time"].max()

    fig, axes = plt.subplots(2, 1, figsize=TIME_SERIES_FIGSIZE, sharex=True)
    ax_p, ax_q = axes

    ax_p.plot(
        df_seg["time"],
        df_seg["P_MW"],
        linewidth=TIME_SERIES_LINEWIDTH,
        label="P [MW]",
        color="#1f77b4",
    )
    ax_p.set_ylabel("P [MW]", fontsize=TIME_SERIES_AXIS_LABEL_FONTSIZE)
    ax_p.tick_params(axis="y", labelsize=TIME_SERIES_TICK_FONTSIZE)
    ax_p.grid(True, alpha=0.35)
    ax_p.legend(loc="upper left", fontsize=TIME_SERIES_LEGEND_FONTSIZE)

    ax_q.plot(
        df_seg["time"],
        df_seg["Q_MVAr"],
        linewidth=TIME_SERIES_LINEWIDTH,
        label="Q [MVAr]",
        color="#ff7f0e",
    )
    ax_q.set_ylabel("Q [MVAr]", fontsize=TIME_SERIES_AXIS_LABEL_FONTSIZE)
    ax_q.tick_params(axis="y", labelsize=TIME_SERIES_TICK_FONTSIZE)
    ax_q.grid(True, alpha=0.35)
    ax_q.legend(loc="upper left", fontsize=TIME_SERIES_LEGEND_FONTSIZE)

    configure_shared_time_axis(ax_q, start_time, end_time)
    save_figure(fig, out_png)


def plot_transformer_segment_qu(
    rtp: str,
    comp_id: str,
    segment_id: int,
    df_seg: pd.DataFrame,
    out_png: Path,
    side_label: str,
):
    """Plot U_pu above Q on aligned axes for the selected transformer side."""
    start_time = df_seg["time"].min()
    end_time = df_seg["time"].max()

    fig, axes = plt.subplots(2, 1, figsize=TIME_SERIES_FIGSIZE, sharex=True)
    ax_u, ax_q = axes

    if df_seg["U_pu"].notna().any():
        ax_u.plot(
            df_seg["time"],
            df_seg["U_pu"],
            linewidth=TIME_SERIES_LINEWIDTH,
            label="U [p.u.]",
            color="#2ca02c",
        )
        configure_voltage_axis(ax_u, df_seg["U_pu"])
        ax_u.legend(loc="upper left", fontsize=TIME_SERIES_LEGEND_FONTSIZE)
    else:
        ax_u.text(
            0.5,
            0.5,
            "U is not available",
            transform=ax_u.transAxes,
            ha="center",
            va="center",
            fontsize=TIME_SERIES_TICK_FONTSIZE,
        )

    ax_u.set_ylabel("U [p.u.]", fontsize=TIME_SERIES_AXIS_LABEL_FONTSIZE)
    ax_u.tick_params(axis="y", labelsize=TIME_SERIES_TICK_FONTSIZE)
    ax_u.grid(True, alpha=0.35)

    ax_q.plot(
        df_seg["time"],
        df_seg["Q_MVAr"],
        linewidth=TIME_SERIES_LINEWIDTH,
        label="Q [MVAr]",
        color="#ff7f0e",
    )
    ax_q.set_ylabel("Q [MVAr]", fontsize=TIME_SERIES_AXIS_LABEL_FONTSIZE)
    ax_q.tick_params(axis="y", labelsize=TIME_SERIES_TICK_FONTSIZE)
    ax_q.grid(True, alpha=0.35)
    ax_q.legend(loc="upper left", fontsize=TIME_SERIES_LEGEND_FONTSIZE)

    configure_shared_time_axis(ax_q, start_time, end_time)
    save_figure(fig, out_png)


def plot_all_transformer_segments(
    rtp: str,
    comp_id: str,
    df_transformer: pd.DataFrame,
    transformer_dir: Path,
    side_label: str,
):
    """
    Create the following plots for one transformer side:
      - P_Q_segment_XXX.png and .svg
      - Q_U_pu_segment_XXX.png and .svg
    """
    transformer_dir.mkdir(parents=True, exist_ok=True)

    df_plot = df_transformer[["time", "P_MW", "Q_MVAr", "U_kV", "U_pu"]].copy()
    df_plot = add_segments(df_plot, "time")

    if SAVE_TRANSFORMER_CSV:
        csv_path = transformer_dir / f"{safe_name(comp_id)}_timeseries.csv"
        df_plot.to_csv(csv_path, index=False, encoding="utf-8-sig")

    segment_summary_rows = []

    for segment_id, df_seg in df_plot.groupby("segment"):
        if len(df_seg) < MIN_POINTS_SEGMENT:
            continue

        start_str = time_name(df_seg["time"].min())
        end_str = time_name(df_seg["time"].max())
        base = f"segment_{int(segment_id):03d}_{start_str}_to_{end_str}"

        png_pq = transformer_dir / f"P_Q_{base}.png"
        png_qu = transformer_dir / f"Q_U_pu_{base}.png"

        plot_transformer_segment_pq(
            rtp, comp_id, int(segment_id), df_seg, png_pq, side_label
        )
        plot_transformer_segment_qu(
            rtp, comp_id, int(segment_id), df_seg, png_qu, side_label
        )

        segment_summary_rows.append({
            "rtp": rtp,
            "component_file_id": comp_id,
            "side": side_label,
            "segment": int(segment_id),
            "start_time": df_seg["time"].min(),
            "end_time": df_seg["time"].max(),
            "n_points": len(df_seg),
            "P_min_MW": df_seg["P_MW"].min(),
            "P_max_MW": df_seg["P_MW"].max(),
            "Q_min_MVAr": df_seg["Q_MVAr"].min(),
            "Q_max_MVAr": df_seg["Q_MVAr"].max(),
            "U_kV_min": df_seg["U_kV"].min(),
            "U_kV_max": df_seg["U_kV"].max(),
            "U_pu_min": df_seg["U_pu"].min(),
            "U_pu_max": df_seg["U_pu"].max(),
            "png_P_Q": str(png_pq),
            "png_Q_U_pu": str(png_qu),
            "svg_P_Q": str(png_pq.with_suffix(".svg")),
            "svg_Q_U_pu": str(png_qu.with_suffix(".svg")),
        })

    if segment_summary_rows:
        pd.DataFrame(segment_summary_rows).to_csv(
            transformer_dir / "segment_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

    return len(segment_summary_rows)


def merge_hv_mv_timeseries(df_hv: pd.DataFrame, df_mv: pd.DataFrame):
    """
    Align HV- and MV-side measurements by time for comparison plots.
    Use an inner join so plots contain only timestamps available on both sides.
    """
    hv = df_hv.rename(columns={
        "P_MW": "P_HV_MW",
        "Q_MVAr": "Q_HV_MVAr",
        "U_kV": "U_HV_kV",
        "U_pu": "U_HV_pu",
    })
    mv = df_mv.rename(columns={
        "P_MW": "P_MV_MW",
        "Q_MVAr": "Q_MV_MVAr",
        "U_kV": "U_MV_kV",
        "U_pu": "U_MV_pu",
    })

    cols_hv = ["time", "P_HV_MW", "Q_HV_MVAr", "U_HV_kV", "U_HV_pu"]
    cols_mv = ["time", "P_MV_MW", "Q_MV_MVAr", "U_MV_kV", "U_MV_pu"]

    df_pair = hv[cols_hv].merge(mv[cols_mv], on="time", how="inner")
    df_pair = df_pair.sort_values("time").reset_index(drop=True)

    df_pair["P_MV_compare_MW"] = (
        MV_POWER_SIGN_FOR_COMPARISON * df_pair["P_MV_MW"]
    )
    df_pair["Q_MV_compare_MVAr"] = (
        MV_POWER_SIGN_FOR_COMPARISON * df_pair["Q_MV_MVAr"]
    )

    return df_pair


def plot_transformer_pair_segment(
    rtp: str,
    transformer_id: str,
    hv_kv: float,
    mv_kv: float,
    segment_id: int,
    df_seg: pd.DataFrame,
    out_png: Path,
):
    """Plot P, Q, and U for the HV and MV sides of a transformer."""
    start_time = df_seg["time"].min()
    end_time = df_seg["time"].max()

    fig, axes = plt.subplots(3, 1, figsize=TIME_SERIES_PAIR_FIGSIZE, sharex=True)

    axes[0].plot(df_seg["time"], df_seg["P_HV_MW"], linewidth=TIME_SERIES_LINEWIDTH, label=f"P HV {hv_kv:g} kV")
    axes[0].plot(df_seg["time"], df_seg["P_MV_compare_MW"], linewidth=TIME_SERIES_LINEWIDTH, label=f"P MV {mv_kv:g} kV × {MV_POWER_SIGN_FOR_COMPARISON:g}")
    axes[0].set_ylabel("P [MW]", fontsize=TIME_SERIES_AXIS_LABEL_FONTSIZE)
    axes[0].grid(True, alpha=0.35)
    axes[0].legend(loc="upper left", fontsize=TIME_SERIES_LEGEND_FONTSIZE)

    axes[1].plot(df_seg["time"], df_seg["Q_HV_MVAr"], linewidth=TIME_SERIES_LINEWIDTH, label=f"Q HV {hv_kv:g} kV")
    axes[1].plot(df_seg["time"], df_seg["Q_MV_compare_MVAr"], linewidth=TIME_SERIES_LINEWIDTH, label=f"Q MV {mv_kv:g} kV × {MV_POWER_SIGN_FOR_COMPARISON:g}")
    axes[1].set_ylabel("Q [MVAr]", fontsize=TIME_SERIES_AXIS_LABEL_FONTSIZE)
    axes[1].grid(True, alpha=0.35)
    axes[1].legend(loc="upper left", fontsize=TIME_SERIES_LEGEND_FONTSIZE)

    if df_seg[["U_HV_pu", "U_MV_pu"]].notna().any().any():
        axes[2].plot(df_seg["time"], df_seg["U_HV_pu"], linewidth=TIME_SERIES_LINEWIDTH, label=f"U HV {hv_kv:g} kV")
        axes[2].plot(df_seg["time"], df_seg["U_MV_pu"], linewidth=TIME_SERIES_LINEWIDTH, label=f"U MV {mv_kv:g} kV")
        configure_voltage_axis(axes[2], df_seg[["U_HV_pu", "U_MV_pu"]])
        axes[2].legend(loc="upper left", fontsize=TIME_SERIES_LEGEND_FONTSIZE)
    else:
        axes[2].text(0.5, 0.5, "U is not available", transform=axes[2].transAxes, ha="center", va="center")

    axes[2].set_ylabel("U [p.u.]", fontsize=TIME_SERIES_AXIS_LABEL_FONTSIZE)
    axes[2].grid(True, alpha=0.35)

    for ax in axes:
        ax.tick_params(axis="both", labelsize=TIME_SERIES_TICK_FONTSIZE)

    configure_shared_time_axis(axes[2], start_time, end_time)
    save_figure(fig, out_png)


def plot_all_transformer_pair_segments(
    rtp: str,
    transformer_id: str,
    hv_kv: float,
    mv_kv: float,
    df_hv: pd.DataFrame,
    df_mv: pd.DataFrame,
    pair_dir: Path,
):
    """Create combined HV/MV comparison plots for continuous segments."""
    pair_dir.mkdir(parents=True, exist_ok=True)

    df_pair = merge_hv_mv_timeseries(df_hv, df_mv)
    if df_pair.empty:
        return 0

    df_pair = add_segments(df_pair, "time")

    if SAVE_TRANSFORMER_CSV:
        df_pair.to_csv(
            pair_dir / f"{safe_name(transformer_id)}_HV_MV_timeseries.csv",
            index=False,
            encoding="utf-8-sig",
        )

    summary_rows = []

    for segment_id, df_seg in df_pair.groupby("segment"):
        if len(df_seg) < MIN_POINTS_SEGMENT:
            continue

        start_str = time_name(df_seg["time"].min())
        end_str = time_name(df_seg["time"].max())
        base = f"segment_{int(segment_id):03d}_{start_str}_to_{end_str}"
        out_png = pair_dir / f"HV_MV_P_Q_U_{base}.png"

        plot_transformer_pair_segment(
            rtp=rtp,
            transformer_id=transformer_id,
            hv_kv=hv_kv,
            mv_kv=mv_kv,
            segment_id=int(segment_id),
            df_seg=df_seg,
            out_png=out_png,
        )

        summary_rows.append({
            "rtp": rtp,
            "transformer_id": transformer_id,
            "segment": int(segment_id),
            "start_time": df_seg["time"].min(),
            "end_time": df_seg["time"].max(),
            "n_common_points": len(df_seg),
            "hv_kv": hv_kv,
            "mv_kv": mv_kv,
            "mv_power_sign_for_comparison": MV_POWER_SIGN_FOR_COMPARISON,
            "P_HV_min_MW": df_seg["P_HV_MW"].min(),
            "P_HV_max_MW": df_seg["P_HV_MW"].max(),
            "P_MV_raw_min_MW": df_seg["P_MV_MW"].min(),
            "P_MV_raw_max_MW": df_seg["P_MV_MW"].max(),
            "Q_HV_min_MVAr": df_seg["Q_HV_MVAr"].min(),
            "Q_HV_max_MVAr": df_seg["Q_HV_MVAr"].max(),
            "Q_MV_raw_min_MVAr": df_seg["Q_MV_MVAr"].min(),
            "Q_MV_raw_max_MVAr": df_seg["Q_MV_MVAr"].max(),
            "U_HV_pu_min": df_seg["U_HV_pu"].min(),
            "U_HV_pu_max": df_seg["U_HV_pu"].max(),
            "U_MV_pu_min": df_seg["U_MV_pu"].min(),
            "U_MV_pu_max": df_seg["U_MV_pu"].max(),
            "png_HV_MV": str(out_png),
            "svg_HV_MV": str(out_png.with_suffix(".svg")),
        })

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            pair_dir / "HV_MV_segment_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

    return len(summary_rows)

# ============================================================
# MAIN PROGRAM
# ============================================================

all_files = sorted(INPUT_DIR.glob("*.parquet"))

if ONLY_110_MV_TRANSFORMERS:
    candidate_files, candidate_stats = build_110_mv_candidates(all_files)
else:
    # Legacy mode: use all 110 kV transformer files without checking for an MV pair.
    candidate_files = []
    for path in all_files:
        meta = parse_meta_from_filename(path)
        if meta is not None and int(meta["voltage_kv"]) == HV_VOLTAGE_KV:
            candidate_files.append((path, None, meta))

    candidate_stats = {
        "parsed_tr_files": None,
        "ignored_files": None,
        "keys_with_tr_files": None,
        "candidate_110_mv": len(candidate_files),
        "excluded_110_without_mv": None,
        "excluded_110_without_mv_rows": [],
        "duplicate_mv_rows": [],
        "ignored_rows": [],
    }

print("=" * 90)
print(f"REACTIVE-POWER DEVIATION FROM cos(phi) = {COSPHI_LIMIT:g} BY RTP")
print("=" * 90)
print(f"Input directory: {INPUT_DIR}")
print(f"Parquet files found: {len(all_files)}")
print(f"Only 110/MV transformers: {ONLY_110_MV_TRANSFORMERS}")
print(f"HV level used for analysis: {HV_VOLTAGE_KV} kV")
print(f"MV levels used for matching: {MV_VOLTAGE_KV_CANDIDATES} kV")
print(f"Transformer files recognized: {candidate_stats['parsed_tr_files']}")
print(f"110/MV candidates found: {len(candidate_files)}")
print(f"110 kV files excluded without an MV pair: {candidate_stats['excluded_110_without_mv']}")
print(f"Output directory: {OUTPUT_DIR}")
print()

if not candidate_files:
    raise RuntimeError(
        "No 110/MV transformer candidates were found. "
        "Check that pairs such as TR_RTP_110_TR1.parquet and TR_RTP_20_TR1.parquet exist "
        "or adjust MV_VOLTAGE_KV_CANDIDATES."
    )

# ============================================================
# BUILD RTP TIME SERIES AND SAVE TRANSFORMER DATA
# ============================================================

rtp_data = {}
rtp_transformers = {}
errors = []
transformer_summary_rows = []

for i, (hv_path, mv_path, meta) in enumerate(candidate_files, start=1):
    mv_text = mv_path.name if mv_path is not None else "without an MV file"
    print(f"[{i}/{len(candidate_files)}] HV: {hv_path.name} | MV: {mv_text}")

    try:
        rtp = meta["rtp"]
        transformer_id = meta["transformer_id"]
        hv_comp_id = meta.get("hv_component_file_id", meta["component_file_id"])
        hv_kv = float(meta["voltage_kv"])

        df_hv = read_transformer_file(hv_path, un_kv=hv_kv)

        if df_hv.empty:
            print("  -> HV side is empty after filtering")
            continue

        df_mv = None
        mv_comp_id = meta.get("mv_component_file_id")
        mv_kv = meta.get("mv_voltage_kv")

        if mv_path is not None and mv_kv is not None:
            try:
                df_mv = read_transformer_file(mv_path, un_kv=float(mv_kv))
                if df_mv.empty:
                    print("  -> WARNING: MV side is empty after filtering")
                    df_mv = None
            except Exception as e_mv:
                print(f"  -> WARNING: error while reading the MV side: {e_mv}")
                errors.append({
                    "file": mv_path.name,
                    "rtp": rtp,
                    "transformer_id": transformer_id,
                    "side": "MV",
                    "error": str(e_mv),
                })
                df_mv = None

        # Store both transformer sides.
        rtp_transformers.setdefault(rtp, {})[hv_comp_id] = {
            "hv_df": df_hv.copy(),
            "mv_df": None if df_mv is None else df_mv.copy(),
            "meta": meta,
            "hv_path": hv_path,
            "mv_path": mv_path,
        }

        # Keep the RTP power-factor analysis on the 110 kV side.
        df_agg = df_hv.rename(columns={
            "P_MW": f"P__{hv_comp_id}",
            "Q_MVAr": f"Q__{hv_comp_id}",
        })
        cols = ["time", f"P__{hv_comp_id}", f"Q__{hv_comp_id}"]

        if rtp not in rtp_data:
            rtp_data[rtp] = df_agg[cols].copy()
        else:
            rtp_data[rtp] = rtp_data[rtp].merge(
                df_agg[cols],
                on="time",
                how="outer",
            )

        n_u_hv = int(df_hv["U_pu"].notna().sum())
        n_u_mv = int(df_mv["U_pu"].notna().sum()) if df_mv is not None else 0

        summary_row = {
            "rtp": rtp,
            "transformer_id": transformer_id,
            "hv_component_file_id": hv_comp_id,
            "hv_file": hv_path.name,
            "hv_voltage_kv": hv_kv,
            "hv_n_points": len(df_hv),
            "hv_n_U_valid": n_u_hv,
            "hv_time_start": df_hv["time"].min(),
            "hv_time_end": df_hv["time"].max(),
            "hv_P_min_MW": df_hv["P_MW"].min(),
            "hv_P_max_MW": df_hv["P_MW"].max(),
            "hv_Q_min_MVAr": df_hv["Q_MVAr"].min(),
            "hv_Q_max_MVAr": df_hv["Q_MVAr"].max(),
            "hv_U_kV_min": df_hv["U_kV"].min(),
            "hv_U_kV_max": df_hv["U_kV"].max(),
            "hv_U_pu_min": df_hv["U_pu"].min(),
            "hv_U_pu_max": df_hv["U_pu"].max(),
            "mv_component_file_id": mv_comp_id,
            "mv_file": None if mv_path is None else mv_path.name,
            "mv_voltage_kv": mv_kv,
            "mv_n_points": 0 if df_mv is None else len(df_mv),
            "mv_n_U_valid": n_u_mv,
            "mv_time_start": pd.NaT if df_mv is None else df_mv["time"].min(),
            "mv_time_end": pd.NaT if df_mv is None else df_mv["time"].max(),
            "mv_P_min_MW": np.nan if df_mv is None else df_mv["P_MW"].min(),
            "mv_P_max_MW": np.nan if df_mv is None else df_mv["P_MW"].max(),
            "mv_Q_min_MVAr": np.nan if df_mv is None else df_mv["Q_MVAr"].min(),
            "mv_Q_max_MVAr": np.nan if df_mv is None else df_mv["Q_MVAr"].max(),
            "mv_U_kV_min": np.nan if df_mv is None else df_mv["U_kV"].min(),
            "mv_U_kV_max": np.nan if df_mv is None else df_mv["U_kV"].max(),
            "mv_U_pu_min": np.nan if df_mv is None else df_mv["U_pu"].min(),
            "mv_U_pu_max": np.nan if df_mv is None else df_mv["U_pu"].max(),
        }
        transformer_summary_rows.append(summary_row)

        if df_mv is None:
            print(
                f"  -> HV OK | RTP = {rtp} | {hv_kv:g} kV | points = {len(df_hv)} | "
                "MV measurements were not read"
            )
        else:
            n_common = len(merge_hv_mv_timeseries(df_hv, df_mv))
            print(
                f"  -> HV/MV OK | RTP = {rtp} | {hv_kv:g}/{float(mv_kv):g} kV | "
                f"HV points = {len(df_hv)} | MV points = {len(df_mv)} | matching timestamps = {n_common}"
            )

    except Exception as e:
        print(f"  -> ERROR: {e}")
        errors.append({
            "file": hv_path.name,
            "rtp": meta.get("rtp"),
            "transformer_id": meta.get("transformer_id"),
            "side": "HV",
            "error": str(e),
        })

# ============================================================
# ANALYZE EACH RTP
# ============================================================

summary_rows = []

for rtp, df_rtp in sorted(rtp_data.items()):
    print()
    print("-" * 90)
    print(f"RTP {rtp}")

    rtp_dir = OUTPUT_DIR / safe_name(rtp)
    rtp_dir.mkdir(parents=True, exist_ok=True)

    p_cols = [c for c in df_rtp.columns if c.startswith("P__")]
    q_cols = [c for c in df_rtp.columns if c.startswith("Q__")]

    if not p_cols or not q_cols:
        print("  -> no P/Q columns")
        continue

    df_rtp = df_rtp.sort_values("time").reset_index(drop=True)

    df_rtp["n_valid_transformers_P"] = df_rtp[p_cols].notna().sum(axis=1)
    df_rtp["n_valid_transformers_Q"] = df_rtp[q_cols].notna().sum(axis=1)

    # Sum P and Q across transformers in the RTP.
    df_rtp["P_MW_sum"] = df_rtp[p_cols].sum(axis=1, skipna=True)
    df_rtp["Q_MVAr_sum"] = df_rtp[q_cols].sum(axis=1, skipna=True)

    # Keep timestamps with at least one valid P and one valid Q measurement.
    df_rtp = df_rtp[
        (df_rtp["n_valid_transformers_P"] > 0) &
        (df_rtp["n_valid_transformers_Q"] > 0)
    ].copy()

    if len(df_rtp) < MIN_POINTS:
        print(f"  -> too few points: {len(df_rtp)}")
        continue

    df_rtp = calculate_dq_from_cosphi_limit(df_rtp, COSPHI_LIMIT)

    n = len(df_rtp)
    n_cap = int((df_rtp["dQ_MVAr"] < 0).sum())
    n_ind = int((df_rtp["dQ_MVAr"] > 0).sum())
    n_ok = int((df_rtp["dQ_MVAr"] == 0).sum())

    pct_cap = 100 * n_cap / n
    pct_ind = 100 * n_ind / n
    pct_ok = 100 * n_ok / n

    dQ_min = df_rtp["dQ_MVAr"].min()
    dQ_max = df_rtp["dQ_MVAr"].max()

    Q_min = df_rtp["Q_MVAr_sum"].min()
    Q_max = df_rtp["Q_MVAr_sum"].max()

    P_min = df_rtp["P_MW_sum"].min()
    P_max = df_rtp["P_MW_sum"].max()

    print(f"  -> points: {n}")
    print(f"  -> transformers: {len(q_cols)}")
    print(f"  -> P_sum min/max: {P_min:.3f} / {P_max:.3f} MW")
    print(f"  -> Q_sum min/max: {Q_min:.3f} / {Q_max:.3f} MVAr")
    print(f"  -> dQ min/max:    {dQ_min:.3f} / {dQ_max:.3f} MVAr")
    print(f"  -> cap/ok/ind:    {pct_cap:.2f}% / {pct_ok:.2f}% / {pct_ind:.2f}%")

    if SAVE_RTP_CSV:
        csv_path = rtp_dir / f"{safe_name(rtp)}_cosphi_095_timeseries.csv"
        df_rtp.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # Store RTP plots in the corresponding RTP directory.
    png_duration = rtp_dir / f"{safe_name(rtp)}_01_dQ_duration_curve_cosphi_095.png"
    plot_duration_curve_together(rtp, df_rtp, png_duration)

    png_share = rtp_dir / f"{safe_name(rtp)}_02_q_status_share_cosphi_095.png"
    plot_q_status_share(rtp, pct_cap, pct_ok, pct_ind, png_share)

    # Transformer plots inside each RTP directory:
    #   HV_110_kV      -> P/Q and Q/U for the HV side
    #   MV_xx_kV       -> P/Q and Q/U for the MV side
    #   HV_MV_COMPARISON -> combined P, Q, and U plot for both sides
    n_hv_segments = 0
    n_mv_segments = 0
    n_pair_segments = 0

    for comp_id, rec in sorted(rtp_transformers.get(rtp, {}).items()):
        meta = rec["meta"]
        transformer_id = meta.get("transformer_id", comp_id)
        hv_kv = float(meta.get("voltage_kv", HV_VOLTAGE_KV))
        mv_kv = meta.get("mv_voltage_kv")

        transformer_dir = rtp_dir / safe_name(comp_id)

        hv_dir = transformer_dir / f"HV_{hv_kv:g}_kV"
        n_hv_segments += plot_all_transformer_segments(
            rtp=rtp,
            comp_id=comp_id,
            df_transformer=rec["hv_df"],
            transformer_dir=hv_dir,
            side_label=f"HV side {hv_kv:g} kV",
        )

        if rec["mv_df"] is not None and mv_kv is not None:
            mv_comp_id = meta.get("mv_component_file_id", f"{transformer_id}_MV")
            mv_dir = transformer_dir / f"MV_{float(mv_kv):g}_kV"
            n_mv_segments += plot_all_transformer_segments(
                rtp=rtp,
                comp_id=mv_comp_id,
                df_transformer=rec["mv_df"],
                transformer_dir=mv_dir,
                side_label=f"MV side {float(mv_kv):g} kV",
            )

            if PLOT_HV_MV_COMPARISON:
                pair_dir = transformer_dir / "HV_MV_COMPARISON"
                n_pair_segments += plot_all_transformer_pair_segments(
                    rtp=rtp,
                    transformer_id=transformer_id,
                    hv_kv=hv_kv,
                    mv_kv=float(mv_kv),
                    df_hv=rec["hv_df"],
                    df_mv=rec["mv_df"],
                    pair_dir=pair_dir,
                )

    print(f"  -> HV segments plotted: {n_hv_segments}")
    print(f"  -> MV segments plotted: {n_mv_segments}")
    print(f"  -> HV/MV comparisons plotted: {n_pair_segments}")
    print(f"  -> RTP directory: {rtp_dir}")

    summary_rows.append({
        "rtp": rtp,
        "n_transformers": len(q_cols),
        "n_points": n,
        "time_start": df_rtp["time"].min(),
        "time_end": df_rtp["time"].max(),

        "P_min_MW": P_min,
        "P_max_MW": P_max,
        "Q_min_MVAr": Q_min,
        "Q_max_MVAr": Q_max,

        "dQ_min_MVAr_cap_peak": dQ_min,
        "dQ_max_MVAr_ind_peak": dQ_max,

        "pct_overcompensated_cap": pct_cap,
        "pct_within_limits": pct_ok,
        "pct_undercompensated_ind": pct_ind,

        "n_overcompensated_cap": n_cap,
        "n_within_limits": n_ok,
        "n_undercompensated_ind": n_ind,

        "cosphi_limit": COSPHI_LIMIT,
        "use_abs_p_for_limits": USE_ABS_P_FOR_LIMITS,
        "rtp_dir": str(rtp_dir),
        "png_duration": str(png_duration),
        "svg_duration": str(png_duration.with_suffix(".svg")),
        "png_status_share": str(png_share),
        "svg_status_share": str(png_share.with_suffix(".svg")),
        "png_share": str(png_share),
    })

# ============================================================
# SAVE SUMMARIES
# ============================================================

summary_df = pd.DataFrame(summary_rows)
summary_path = OUTPUT_DIR / "RTP_cosphi_095_summary.csv"

if not summary_df.empty:
    summary_df = summary_df.sort_values(
        by=["pct_overcompensated_cap", "dQ_min_MVAr_cap_peak"],
        ascending=[False, True],
    )

summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

transformer_summary_df = pd.DataFrame(transformer_summary_rows)
transformer_summary_path = OUTPUT_DIR / "transformer_summary.csv"

if not transformer_summary_df.empty:
    transformer_summary_df = transformer_summary_df.sort_values(["rtp", "hv_component_file_id"])

transformer_summary_df.to_csv(transformer_summary_path, index=False, encoding="utf-8-sig")


# Save the candidate audit so excluded 110 kV transformers can be reviewed,
# including those without a matching MV file.
if ONLY_110_MV_TRANSFORMERS and candidate_stats.get("excluded_110_without_mv_rows"):
    excluded_110_path = OUTPUT_DIR / "excluded_110_without_MV_pair.csv"
    pd.DataFrame(candidate_stats["excluded_110_without_mv_rows"]).to_csv(
        excluded_110_path,
        index=False,
        encoding="utf-8-sig",
    )

if ONLY_110_MV_TRANSFORMERS and candidate_stats.get("ignored_rows"):
    ignored_files_path = OUTPUT_DIR / "ignored_TR_parser_files.csv"
    pd.DataFrame(candidate_stats["ignored_rows"]).to_csv(
        ignored_files_path,
        index=False,
        encoding="utf-8-sig",
    )

if ONLY_110_MV_TRANSFORMERS and candidate_stats.get("duplicate_mv_rows"):
    duplicate_mv_path = OUTPUT_DIR / "duplicate_MV_pairs.csv"
    pd.DataFrame(candidate_stats["duplicate_mv_rows"]).to_csv(
        duplicate_mv_path,
        index=False,
        encoding="utf-8-sig",
    )

if errors:
    errors_df = pd.DataFrame(errors)
    errors_path = OUTPUT_DIR / "RTP_cosphi_095_errors.csv"
    errors_df.to_csv(errors_path, index=False, encoding="utf-8-sig")

print()
print("=" * 90)
print("DONE")
print("=" * 90)
print(f"RTPs analyzed: {len(summary_df)}")
print(f"RTP summary:     {summary_path}")
print(f"Transformer summary: {transformer_summary_path}")
print(f"Output directory:    {OUTPUT_DIR}")
if ONLY_110_MV_TRANSFORMERS:
    print(f"110 kV files excluded without an MV pair: {candidate_stats['excluded_110_without_mv']}")
    if candidate_stats.get("excluded_110_without_mv_rows"):
        print(f"Exclusion audit: {excluded_110_path}")

if not summary_df.empty:
    print()
    print("TOP 20 RTPs BY OVERCOMPENSATION SHARE / CAP")
    print(
        summary_df[
            [
                "rtp",
                "n_transformers",
                "pct_overcompensated_cap",
                "pct_within_limits",
                "pct_undercompensated_ind",
                "dQ_min_MVAr_cap_peak",
                "dQ_max_MVAr_ind_peak",
                "P_min_MW",
                "P_max_MW",
                "Q_min_MVAr",
                "Q_max_MVAr",
                "rtp_dir",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )
