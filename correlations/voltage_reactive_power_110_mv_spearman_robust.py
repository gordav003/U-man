from __future__ import annotations

"""Robustna primerjava Spearmanove korelacije 15-minutnih sprememb U in Q.

Skripta ohrani strogo identifikacijo transformatorjev 110/SN iz izvorne
skripte, vendar pri agregaciji ohrani tudi identiteto uporabljenih trafov.
Izracuna originalno in robustno metodo na istih oseh ter opozori na spremembe
nabora meritev. Predznak Q se nikjer ne spreminja: Q < 0 je kapacitivna in
Q > 0 induktivna jalova moc.

Privzeta metodologija je:

* PAIR_VALIDATION_MODE = "same_set": za dU oziroma dQ mora biti nabor trafov
  pri t in t+15 enak;
* MEASUREMENT_SET_MODE = "common_uq": v robustni agregat se trafo vkljuci le,
  ce sta v istem timestampu veljavni obe meritvi U in Q.

Strozji ``all_expected`` je na voljo kot argument, vendar ni privzet, ker
lahko ze en obcasno manjkajoc kanal odstrani vse pare RTP in povzroci mocno
selekcijo podatkov. Manjkajoce vrednosti se ne interpolirajo, segmenti se ne
zdruzujejo in uporabljajo se samo tocni 15-minutni pari.

Zagon iz korena projekta:

    python -m correlations.voltage_reactive_power_110_mv_spearman_robust
"""

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import polars as pl
from scipy.stats import ConstantInputWarning, spearmanr


HV_VOLTAGE_KV = 110
VALID_U_MIN_PU = 0.5
VALID_U_MAX_PU = 1.5
DELTA_MINUTES = 15
DEFAULT_MIN_PAIR_POINTS = 20
DEFAULT_MAX_GAP_MINUTES = 15
DEFAULT_MIN_SEGMENT_POINTS = 20

# Metodolosko priporocena privzeta izbira; obe konstanti je mogoce preglasiti
# z argumentoma ukazne vrstice.
PAIR_VALIDATION_MODE = "same_set"
MEASUREMENT_SET_MODE = "common_uq"

FREQUENT_TRANSITION_RATE = 0.05
HIGH_REJECTION_RATE = 0.50
NEAR_CONSTANT_REL_TOL = 1e-8
NEAR_CONSTANT_ABS_TOL = 1e-9


@dataclass(frozen=True)
class ContinuousSegment:
    number: int
    start: datetime
    end: datetime
    n_timestamps: int

    @property
    def tag(self) -> str:
        return (
            f"segment_{self.number:03d}_"
            f"{self.start:%Y%m%d_%H%M}_do_{self.end:%Y%m%d_%H%M}"
        )

    @property
    def label(self) -> str:
        return f"{self.start:%Y-%m-%d %H:%M} do {self.end:%Y-%m-%d %H:%M}"


@dataclass
class LoadedData:
    busbars: pl.DataFrame
    transformer_values: pl.DataFrame
    metadata: pl.DataFrame
    duplicate_diagnostics: pl.DataFrame


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_input_path() -> Path:
    return (
        project_root()
        / "Uman meritve"
        / "Pridobljeno in urejeno"
        / "urejeno"
        / "Uman_parquet"
        / "transformers_wide.parquet"
    )


def default_output_dir() -> Path:
    return (
        project_root()
        / "Uman meritve"
        / "korelacija_U_Q_110_SN_spearman_robust"
        / "zvezni_segmenti"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Primerja originalno in robustno Spearmanovo korelacijo dU-dQ."
    )
    parser.add_argument("--input", type=Path, default=default_input_path())
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument(
        "--pair-validation-mode",
        choices=("same_set", "all_expected"),
        default=PAIR_VALIDATION_MODE,
        help="same_set (privzeto) ali strozji all_expected.",
    )
    parser.add_argument(
        "--measurement-set-mode",
        choices=("common_uq", "separate_u_q"),
        default=MEASUREMENT_SET_MODE,
        help="Skupni veljavni UQ nabor (privzeto) ali locena nabora U in Q.",
    )
    parser.add_argument(
        "--min-pair-points", type=int, default=DEFAULT_MIN_PAIR_POINTS
    )
    parser.add_argument(
        "--max-gap-minutes", type=int, default=DEFAULT_MAX_GAP_MINUTES
    )
    parser.add_argument(
        "--min-segment-points", type=int, default=DEFAULT_MIN_SEGMENT_POINTS
    )
    parser.add_argument("--annotate-max-busbars", type=int, default=45)
    return parser.parse_args()


def validate_schema(parquet_path: Path) -> None:
    required = {"time", "component_id", "napetost_kv", "lokacija_od", "objekt", "Q", "U"}
    available = set(pl.scan_parquet(parquet_path).collect_schema().names())
    missing = sorted(required - available)
    if missing:
        raise ValueError("V vhodnem Parquetu manjkajo stolpci: " + ", ".join(missing))


def load_110_sn_data(parquet_path: Path, measurement_set_mode: str) -> LoadedData:
    """Nalozi stroge trafe 110/SN in sestavi originalne ter robustne agregate."""
    source = (
        pl.scan_parquet(parquet_path)
        .filter(pl.col("lokacija_od").is_not_null())
        .filter(pl.col("lokacija_od").str.strip_chars() != "")
        .with_columns(
            pl.col("objekt").fill_null(pl.col("component_id")).alias("fizicni_trafo")
        )
    )
    levels = source.select("lokacija_od", "fizicni_trafo", "napetost_kv").unique()
    strict_keys = (
        levels.group_by(["lokacija_od", "fizicni_trafo"])
        .agg(
            pl.col("napetost_kv").max().alias("najvisji_nivo_kv"),
            (pl.col("napetost_kv") < HV_VOLTAGE_KV).any().alias("ima_SN_stran"),
            pl.col("napetost_kv").sort().unique().alias("nivoji_kv"),
        )
        .filter(pl.col("najvisji_nivo_kv") == HV_VOLTAGE_KV)
        .filter(pl.col("ima_SN_stran"))
    )
    strict_110 = source.filter(pl.col("napetost_kv") == HV_VOLTAGE_KV).join(
        strict_keys.select("lokacija_od", "fizicni_trafo", "nivoji_kv"),
        on=["lokacija_od", "fizicni_trafo"],
        how="inner",
    )

    valid_u = (
        pl.col("U").is_not_null()
        & pl.col("U").is_finite()
        & (pl.col("U") / float(HV_VOLTAGE_KV)).is_between(
            VALID_U_MIN_PU, VALID_U_MAX_PU, closed="both"
        )
    )
    valid_q = pl.col("Q").is_not_null() & pl.col("Q").is_finite()

    transformer_values_lf = strict_110.group_by(
        ["time", "lokacija_od", "fizicni_trafo"]
    ).agg(
        pl.col("U").filter(valid_u).median().alias("U_trafo_kV"),
        pl.col("Q").filter(valid_q).median().alias("Q_trafo_MVAr"),
        pl.len().alias("n_110_vrstic"),
        pl.col("component_id").n_unique().alias("n_110_component_id"),
    )
    transformer_values = transformer_values_lf.with_columns(
        pl.col("U_trafo_kV").is_not_null().alias("valid_U"),
        pl.col("Q_trafo_MVAr").is_not_null().alias("valid_Q"),
        (
            pl.col("U_trafo_kV").is_not_null()
            & pl.col("Q_trafo_MVAr").is_not_null()
        ).alias("valid_UQ"),
    ).collect()

    duplicate_diagnostics = (
        transformer_values.filter(
            (pl.col("n_110_vrstic") > 1) | (pl.col("n_110_component_id") > 1)
        )
        .select(
            "time",
            "lokacija_od",
            "fizicni_trafo",
            "n_110_vrstic",
            "n_110_component_id",
        )
        .sort(["lokacija_od", "fizicni_trafo", "time"])
    )

    eligible_locations = (
        transformer_values.group_by("lokacija_od")
        .agg(
            pl.col("valid_U").sum().alias("n_veljavnih_U"),
            pl.col("valid_Q").sum().alias("n_veljavnih_Q"),
        )
        .filter((pl.col("n_veljavnih_U") > 0) & (pl.col("n_veljavnih_Q") > 0))
        .select("lokacija_od")
    )

    strict_keys_collected = strict_keys.collect().join(
        eligible_locations, on="lokacija_od", how="inner"
    )
    metadata = (
        strict_keys_collected.with_columns(
            pl.col("nivoji_kv")
            .list.eval(pl.element().cast(pl.String))
            .list.join(",")
            .alias("nivoji_trafo_kV")
        )
        .group_by("lokacija_od")
        .agg(
            pl.col("fizicni_trafo").sort().alias("expected_transformer_set"),
            pl.col("fizicni_trafo").n_unique().alias("n_expected_transformers"),
            pl.col("fizicni_trafo").sort().str.join(";").alias("transformatorji"),
            pl.col("nivoji_trafo_kV").sort().str.join(";").alias("nivoji_transformatorjev_kV"),
        )
        .with_columns(
            (pl.col("lokacija_od") + pl.lit(" | 110 kV")).alias("zbiralka")
        )
        .sort("lokacija_od")
    )

    tv = transformer_values.join(eligible_locations, on="lokacija_od", how="inner")
    robust_u_valid = pl.col("valid_UQ") if measurement_set_mode == "common_uq" else pl.col("valid_U")
    robust_q_valid = pl.col("valid_UQ") if measurement_set_mode == "common_uq" else pl.col("valid_Q")
    busbars = (
        tv.group_by(["time", "lokacija_od"])
        .agg(
            # Originalna metoda je namerno enaka izvorni kodi.
            pl.col("U_trafo_kV").median().alias("U_original_kV"),
            pl.col("Q_trafo_MVAr").sum().alias("Q_original_MVAr"),
            pl.col("valid_U").sum().alias("n_trafov_U_original"),
            pl.col("valid_Q").sum().alias("n_trafov_Q_original"),
            pl.col("fizicni_trafo").filter(pl.col("valid_U")).sort().alias("trafoti_U_original_set"),
            pl.col("fizicni_trafo").filter(pl.col("valid_Q")).sort().alias("trafoti_Q_original_set"),
            # Robustna metoda uporablja skupni UQ nabor ali izrecno izbrana locena nabora.
            pl.col("U_trafo_kV").filter(robust_u_valid).median().alias("U_robust_kV"),
            pl.col("Q_trafo_MVAr").filter(robust_q_valid).sum().alias("Q_robust_MVAr"),
            robust_u_valid.sum().alias("n_trafov_U_robust"),
            robust_q_valid.sum().alias("n_trafov_Q_robust"),
            pl.col("fizicni_trafo").filter(robust_u_valid).sort().alias("trafoti_U_robust_set"),
            pl.col("fizicni_trafo").filter(robust_q_valid).sort().alias("trafoti_Q_robust_set"),
            pl.col("fizicni_trafo").sort().unique().alias("structural_transformer_set"),
        )
        .join(
            metadata.select("lokacija_od", "zbiralka", "expected_transformer_set", "n_expected_transformers"),
            on="lokacija_od",
            how="left",
        )
        .with_columns(
            pl.when(pl.col("n_trafov_Q_original") > 0)
            .then(pl.col("Q_original_MVAr"))
            .otherwise(None)
            .alias("Q_original_MVAr"),
            pl.when(pl.col("trafoti_Q_robust_set").list.len() > 0)
            .then(pl.col("Q_robust_MVAr"))
            .otherwise(None)
            .alias("Q_robust_MVAr"),
        )
        .sort(["lokacija_od", "time"])
    )
    return LoadedData(busbars, transformer_values, metadata, duplicate_diagnostics)


def find_continuous_segments(
    busbars: pl.DataFrame, max_gap_minutes: int, min_segment_points: int
) -> list[ContinuousSegment]:
    times = busbars.select("time").unique().sort("time").get_column("time").to_list()
    if not times:
        return []
    maximum_gap = timedelta(minutes=max_gap_minutes)
    raw: list[list[datetime]] = [[times[0]]]
    for previous, current in zip(times, times[1:]):
        if current - previous > maximum_gap:
            raw.append([])
        raw[-1].append(current)
    result: list[ContinuousSegment] = []
    for segment_times in raw:
        if len(segment_times) >= min_segment_points:
            result.append(
                ContinuousSegment(
                    len(result) + 1,
                    segment_times[0],
                    segment_times[-1],
                    len(segment_times),
                )
            )
    return result


def rows_in_segment(frame: pl.DataFrame, segment: ContinuousSegment) -> pl.DataFrame:
    return frame.filter(pl.col("time").is_between(segment.start, segment.end, closed="both"))


def exact_15_minute_changes(busbars: pl.DataFrame, validation_mode: str) -> pl.DataFrame:
    """Sestavi tocne pare in loceno oznaci originalno ter robustno veljavnost."""
    value_cols = [
        "lokacija_od", "zbiralka", "U_original_kV", "Q_original_MVAr",
        "U_robust_kV", "Q_robust_MVAr", "trafoti_U_robust_set",
        "trafoti_Q_robust_set", "structural_transformer_set",
        "expected_transformer_set", "n_expected_transformers",
    ]
    current = busbars.select("time", *value_cols).rename(
        {name: f"{name}_t" for name in value_cols if name not in {"lokacija_od", "zbiralka"}}
    )
    future = (
        busbars.select("time", *value_cols)
        .with_columns((pl.col("time") - pl.duration(minutes=DELTA_MINUTES)).alias("time"))
        .rename({name: f"{name}_f" for name in value_cols if name not in {"lokacija_od", "zbiralka"}})
    )
    paired = current.join(future, on=["time", "lokacija_od", "zbiralka"], how="inner")
    same_u = pl.col("trafoti_U_robust_set_t") == pl.col("trafoti_U_robust_set_f")
    same_q = pl.col("trafoti_Q_robust_set_t") == pl.col("trafoti_Q_robust_set_f")
    nonempty_u = pl.col("trafoti_U_robust_set_t").list.len() > 0
    nonempty_q = pl.col("trafoti_Q_robust_set_t").list.len() > 0
    if validation_mode == "same_set":
        valid_du_set = same_u & nonempty_u
        valid_dq_set = same_q & nonempty_q
    else:
        valid_du_set = (
            (pl.col("trafoti_U_robust_set_t") == pl.col("expected_transformer_set_t"))
            & (pl.col("trafoti_U_robust_set_f") == pl.col("expected_transformer_set_f"))
        )
        valid_dq_set = (
            (pl.col("trafoti_Q_robust_set_t") == pl.col("expected_transformer_set_t"))
            & (pl.col("trafoti_Q_robust_set_f") == pl.col("expected_transformer_set_f"))
        )
    return (
        paired.with_columns(
            (pl.col("U_original_kV_f") - pl.col("U_original_kV_t")).alias("dU_original_kV"),
            (pl.col("Q_original_MVAr_f") - pl.col("Q_original_MVAr_t")).alias("dQ_original_MVAr"),
            same_u.alias("same_U_set"),
            same_q.alias("same_Q_set"),
            (
                (pl.col("trafoti_U_robust_set_t") != pl.col("expected_transformer_set_t"))
                | (pl.col("trafoti_U_robust_set_f") != pl.col("expected_transformer_set_f"))
                | (pl.col("trafoti_Q_robust_set_t") != pl.col("expected_transformer_set_t"))
                | (pl.col("trafoti_Q_robust_set_f") != pl.col("expected_transformer_set_f"))
            ).alias("incomplete_expected_set"),
            valid_du_set.alias("valid_dU_set"),
            valid_dq_set.alias("valid_dQ_set"),
        )
        .with_columns(
            pl.when(pl.col("valid_dU_set"))
            .then(pl.col("U_robust_kV_f") - pl.col("U_robust_kV_t"))
            .otherwise(None)
            .alias("dU_robust_kV"),
            pl.when(pl.col("valid_dQ_set"))
            .then(pl.col("Q_robust_MVAr_f") - pl.col("Q_robust_MVAr_t"))
            .otherwise(None)
            .alias("dQ_robust_MVAr"),
        )
        .with_columns(
            (pl.col("dU_original_kV").is_not_null() & pl.col("dQ_original_MVAr").is_not_null()).alias("valid_pair_original"),
            (pl.col("dU_robust_kV").is_not_null() & pl.col("dQ_robust_MVAr").is_not_null()).alias("valid_pair_robust"),
            (~pl.col("same_U_set") | ~pl.col("same_Q_set")).alias("transformer_set_changed"),
        )
        .sort(["time", "lokacija_od"])
    )


def eligible_busbars(changes: pl.DataFrame, metadata: pl.DataFrame, minimum: int) -> tuple[list[str], pl.DataFrame]:
    availability = changes.group_by("zbiralka").agg(
        pl.col("dU_original_kV").is_not_null().sum().alias("n_du_original"),
        pl.col("dQ_original_MVAr").is_not_null().sum().alias("n_dq_original"),
        pl.col("dU_robust_kV").is_not_null().sum().alias("n_du_robust"),
        pl.col("dQ_robust_MVAr").is_not_null().sum().alias("n_dq_robust"),
    ).filter(
        ((pl.col("n_du_original") >= minimum) & (pl.col("n_dq_original") >= minimum))
        | ((pl.col("n_du_robust") >= minimum) & (pl.col("n_dq_robust") >= minimum))
    )
    selected = metadata.join(availability, on="zbiralka", how="inner").sort("lokacija_od")
    return selected.get_column("zbiralka").to_list(), selected


def wide_values(changes: pl.DataFrame, names: list[str], value_column: str) -> np.ndarray:
    wide = changes.pivot(index="time", on="zbiralka", values=value_column, aggregate_function="first").sort("time")
    for name in names:
        if name not in wide.columns:
            wide = wide.with_columns(pl.lit(None, dtype=pl.Float64).alias(name))
    wide = wide.select(names)
    return np.column_stack([wide.get_column(name).to_numpy() for name in names]).astype(float, copy=False)


def spearman_cross_matrix(du: np.ndarray, dq: np.ndarray, minimum: int) -> tuple[np.ndarray, np.ndarray]:
    valid_u = np.isfinite(du)
    valid_q = np.isfinite(dq)
    counts = valid_u.astype(np.int32).T @ valid_q.astype(np.int32)
    n = du.shape[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        result = spearmanr(np.column_stack([du, dq]), axis=0, nan_policy="omit")
    full = np.asarray(result.statistic, dtype=float)
    correlation = full.reshape(1, 1) if n == 1 else full[:n, n:]
    correlation[counts < minimum] = np.nan
    return correlation, counts


def write_matrix_csv(path: Path, names: list[str], matrix: np.ndarray, integer: bool = False) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["dU_zbiralka", *[f"dQ: {name}" for name in names]])
        for name, row in zip(names, matrix):
            values = (
                [str(int(value)) for value in row]
                if integer
                else ["" if not np.isfinite(value) else f"{value:.6f}" for value in row]
            )
            writer.writerow([f"dU: {name}", *values])


def draw_heatmap(
    matrix: np.ndarray,
    names: list[str],
    png_path: Path,
    svg_path: Path,
    annotate_max: int,
    title: str,
    subtitle: str,
    colorbar_label: str,
) -> None:
    n = len(names)
    size = max(18.0, min(54.0, 0.31 * n))
    font = max(3.2, min(7.0, 520.0 / max(n, 1)))
    cmap = LinearSegmentedColormap.from_list("blue_white_orange", ["#2463A8", "#F7F8FA", "#D97706"])
    cmap.set_bad("#D9DDE3")
    fig, ax = plt.subplots(figsize=(size, size), dpi=180)
    image = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=-1, vmax=1, interpolation="nearest", aspect="equal")
    ax.set_title(title, fontsize=max(12, font * 2.2), color="#20252B", pad=18)
    ax.text(0, 1.012, subtitle, transform=ax.transAxes, fontsize=max(8, font * 1.45), color="#5B6470", va="bottom")
    positions = np.arange(n)
    ax.set_xticks(positions)
    ax.set_yticks(positions)
    ax.set_xticklabels(names, rotation=90, fontsize=font)
    ax.set_yticklabels(names, fontsize=font)
    ax.tick_params(axis="both", length=0, pad=2)
    if n <= annotate_max:
        for row in range(n):
            for column in range(n):
                value = matrix[row, column]
                if np.isfinite(value):
                    ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=max(4.5, font * 0.82), color="white" if abs(value) >= 0.62 else "#20252B")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.012)
    colorbar.set_label(colorbar_label, color="#20252B")
    colorbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xlabel("ΔQ / MVAr")
    ax.set_ylabel("ΔU / kV")
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)


def validation_diagnostics(changes: pl.DataFrame, mode: str) -> pl.DataFrame:
    return (
        changes.group_by(["zbiralka", "lokacija_od", "n_expected_transformers_t"])
        .agg(
            pl.len().alias("n_all_15min_pairs"),
            pl.col("valid_pair_original").sum().alias("n_valid_pairs_original"),
            pl.col("valid_pair_robust").sum().alias("n_valid_pairs"),
            ((~pl.col("valid_pair_robust")) & pl.col("transformer_set_changed")).sum().alias("n_rejected_transformer_set_change"),
            ((~pl.col("valid_pair_robust")) & pl.col("incomplete_expected_set") & ~pl.col("transformer_set_changed")).sum().alias("n_rejected_incomplete_expected"),
            ((~pl.col("valid_pair_robust")) & ~pl.col("transformer_set_changed") & ~pl.col("incomplete_expected_set")).sum().alias("n_rejected_missing_measurement"),
        )
        .rename({"n_expected_transformers_t": "n_expected_transformers"})
        .with_columns(
            (pl.col("n_all_15min_pairs") - pl.col("n_valid_pairs")).alias("n_rejected_total"),
            (100.0 * (pl.col("n_all_15min_pairs") - pl.col("n_valid_pairs")) / pl.col("n_all_15min_pairs")).alias("pct_rejected"),
            (100.0 * pl.col("n_rejected_transformer_set_change") / pl.col("n_all_15min_pairs")).alias("pct_rejected_set_change"),
            pl.lit(mode).alias("validation_mode"),
        )
        .sort("lokacija_od")
    )


def same_busbar_comparison(
    metadata: pl.DataFrame,
    original: np.ndarray,
    robust: np.ndarray,
    original_counts: np.ndarray,
    robust_counts: np.ndarray,
) -> pl.DataFrame:
    orig_diag = np.diag(original)
    robust_diag = np.diag(robust)
    difference = robust_diag - orig_diag
    return (
        metadata.select("zbiralka", "lokacija_od", "n_expected_transformers", "transformatorji")
        .with_columns(
            pl.Series("spearman_original", orig_diag, nan_to_null=True),
            pl.Series("spearman_robust", robust_diag, nan_to_null=True),
            pl.Series("delta_rho_robust_minus_original", difference, nan_to_null=True),
            pl.Series("n_pairs_original", np.diag(original_counts)),
            pl.Series("n_pairs_robust", np.diag(robust_counts)),
        )
        .with_columns(pl.col("delta_rho_robust_minus_original").abs().alias("abs_delta_rho"))
        .sort("abs_delta_rho", descending=True, nulls_last=True)
    )


def set_to_text(values: object) -> str:
    return ";".join(values or [])


def quality_diagnostics(
    segment: ContinuousSegment,
    segment_busbars: pl.DataFrame,
    changes: pl.DataFrame,
    metadata: pl.DataFrame,
    validation: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Vrne opozorila in diagnostiko pojavljanja posameznih trafov."""
    warning_rows: list[dict[str, object]] = []
    appearance_rows: list[dict[str, object]] = []
    expected_by_location = {
        row["lokacija_od"]: row["expected_transformer_set"]
        for row in metadata.iter_rows(named=True)
    }

    for location_frame in segment_busbars.partition_by("lokacija_od", maintain_order=True):
        location = location_frame["lokacija_od"][0]
        zbiralka = location_frame["zbiralka"][0]
        structural_counts = location_frame["structural_transformer_set"].list.len()
        if structural_counts.n_unique() > 1:
            warning_rows.append({
                "segment": segment.tag, "zbiralka": zbiralka, "lokacija_od": location,
                "severity": "HIGH", "check": "structural_transformer_count_changes",
                "message": f"Stevilo strukturno prisotnih trafov se spreminja ({structural_counts.min()}..{structural_counts.max()}); preveri izvorne vrstice ali spremembo topologije.",
            })
        for column, unit in (("U_robust_kV", "kV"), ("Q_robust_MVAr", "MVAr")):
            values = location_frame[column].drop_nulls().to_numpy()
            if values.size >= 2:
                value_range = float(np.max(values) - np.min(values))
                scale = max(float(np.median(np.abs(values))), 1.0)
                if value_range == 0.0 or value_range <= max(NEAR_CONSTANT_ABS_TOL, NEAR_CONSTANT_REL_TOL * scale):
                    warning_rows.append({
                        "segment": segment.tag, "zbiralka": zbiralka, "lokacija_od": location,
                        "severity": "MEDIUM", "check": f"constant_or_near_constant_{column}",
                        "message": f"Robustni {column} je konstanten ali skoraj konstanten; razpon={value_range:.6g} {unit}.",
                    })

        location_changes = changes.filter(pl.col("lokacija_od") == location)
        n_pairs = location_changes.height
        for transformer in expected_by_location.get(location, []):
            transitions = 0
            present_t = 0
            present_f = 0
            structural_transitions = 0
            for row in location_changes.select(
                "trafoti_U_robust_set_t", "trafoti_U_robust_set_f",
                "structural_transformer_set_t", "structural_transformer_set_f",
            ).iter_rows(named=True):
                in_t = transformer in (row["trafoti_U_robust_set_t"] or [])
                in_f = transformer in (row["trafoti_U_robust_set_f"] or [])
                present_t += int(in_t)
                present_f += int(in_f)
                transitions += int(in_t != in_f)
                structural_transitions += int(
                    (transformer in (row["structural_transformer_set_t"] or []))
                    != (transformer in (row["structural_transformer_set_f"] or []))
                )
            denominator = max(2 * n_pairs, 1)
            availability = (present_t + present_f) / denominator
            transition_rate = transitions / max(n_pairs, 1)
            appearance_rows.append({
                "segment": segment.tag, "zbiralka": zbiralka, "lokacija_od": location,
                "fizicni_trafo": transformer, "n_exact_pairs": n_pairs,
                "valid_measurement_availability_pct": 100.0 * availability,
                "n_valid_set_transitions": transitions,
                "valid_set_transition_rate_pct": 100.0 * transition_rate,
                "n_structural_transitions": structural_transitions,
            })
            if transitions > 0 and transition_rate >= FREQUENT_TRANSITION_RATE:
                warning_rows.append({
                    "segment": segment.tag, "zbiralka": zbiralka, "lokacija_od": location,
                    "severity": "MEDIUM", "check": "frequent_transformer_appearance_changes",
                    "message": f"{transformer}: {transitions}/{n_pairs} ({100*transition_rate:.1f} %) prehodov veljavnega nabora.",
                })

    for row in validation.iter_rows(named=True):
        if row["pct_rejected"] >= 100.0 * HIGH_REJECTION_RATE:
            warning_rows.append({
                "segment": segment.tag, "zbiralka": row["zbiralka"], "lokacija_od": row["lokacija_od"],
                "severity": "HIGH", "check": "high_robust_pair_loss",
                "message": f"Robustni filter odstrani {row['pct_rejected']:.1f} % tocnih 15-minutnih parov.",
            })

    warning_schema = {
        "segment": pl.String, "zbiralka": pl.String, "lokacija_od": pl.String,
        "severity": pl.String, "check": pl.String, "message": pl.String,
    }
    appearance_schema = {
        "segment": pl.String, "zbiralka": pl.String, "lokacija_od": pl.String,
        "fizicni_trafo": pl.String, "n_exact_pairs": pl.Int64,
        "valid_measurement_availability_pct": pl.Float64,
        "n_valid_set_transitions": pl.Int64, "valid_set_transition_rate_pct": pl.Float64,
        "n_structural_transitions": pl.Int64,
    }
    return (
        pl.DataFrame(warning_rows, schema=warning_schema),
        pl.DataFrame(appearance_rows, schema=appearance_schema),
    )


def write_segments_index(path: Path, segments: list[ContinuousSegment]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["segment", "zacetek", "konec", "stevilo_casovnih_tock"])
        for segment in segments:
            writer.writerow([segment.tag, segment.start.isoformat(sep=" "), segment.end.isoformat(sep=" "), segment.n_timestamps])


def write_public_metadata(path: Path, metadata: pl.DataFrame) -> None:
    metadata.with_columns(
        pl.col("expected_transformer_set").list.join(";").alias("expected_transformer_set")
    ).write_csv(path, separator=";", include_bom=True)


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Vhodna datoteka ne obstaja: {input_path}")
    if args.min_pair_points < 2 or args.min_segment_points < 2:
        raise ValueError("Minimalno stevilo parov in tock segmenta mora biti vsaj 2.")
    if args.max_gap_minutes < 1 or args.annotate_max_busbars < 0:
        raise ValueError("Neveljaven max-gap-minutes ali annotate-max-busbars.")

    validate_schema(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    loaded = load_110_sn_data(input_path, args.measurement_set_mode)
    if loaded.metadata.is_empty():
        raise ValueError("Ni strogih zbiralk 110/SN z veljavnimi meritvami U in Q.")

    segments = find_continuous_segments(loaded.busbars, args.max_gap_minutes, args.min_segment_points)
    if not segments:
        raise ValueError("Ni dovolj dolgega zveznega segmenta za analizo.")
    write_segments_index(output_dir / "segmenti.csv", segments)
    write_public_metadata(output_dir / "zbiralke_110_SN.csv", loaded.metadata)
    loaded.duplicate_diagnostics.write_csv(
        output_dir / "podvojeni_zapisi_transformatorjev.csv", separator=";", include_bom=True
    )

    all_validation: list[pl.DataFrame] = []
    all_warnings: list[pl.DataFrame] = []
    all_appearance: list[pl.DataFrame] = []
    for segment in segments:
        segment_dir = output_dir / segment.tag
        segment_dir.mkdir(parents=True, exist_ok=True)
        segment_busbars = rows_in_segment(loaded.busbars, segment)
        changes = exact_15_minute_changes(segment_busbars, args.pair_validation_mode)
        names, selected_metadata = eligible_busbars(changes, loaded.metadata, args.min_pair_points)
        validation = validation_diagnostics(changes, args.pair_validation_mode).with_columns(
            pl.lit(segment.tag).alias("segment")
        )
        quality_warnings, appearance = quality_diagnostics(
            segment, segment_busbars, changes, loaded.metadata, validation
        )
        all_validation.append(validation)
        all_warnings.append(quality_warnings)
        all_appearance.append(appearance)
        validation.write_csv(segment_dir / "diagnostika_validacije_15min_parov.csv", separator=";", include_bom=True)
        quality_warnings.write_csv(segment_dir / "opozorila_kakovosti.csv", separator=";", include_bom=True)
        appearance.write_csv(segment_dir / "diagnostika_pojavljanja_transformatorjev.csv", separator=";", include_bom=True)

        if not names:
            (segment_dir / "izracun_povzetek.txt").write_text(
                "V segmentu ni lokacij z dovolj pari niti po originalni niti po robustni metodi.\n",
                encoding="utf-8",
            )
            print(f"Segment {segment.number}/{len(segments)} preskocen: premalo parov.")
            continue
        selected_changes = changes.filter(pl.col("zbiralka").is_in(names))
        du_original = wide_values(selected_changes, names, "dU_original_kV")
        dq_original = wide_values(selected_changes, names, "dQ_original_MVAr")
        du_robust = wide_values(selected_changes, names, "dU_robust_kV")
        dq_robust = wide_values(selected_changes, names, "dQ_robust_MVAr")
        corr_original, count_original = spearman_cross_matrix(du_original, dq_original, args.min_pair_points)
        corr_robust, count_robust = spearman_cross_matrix(du_robust, dq_robust, args.min_pair_points)
        difference = corr_robust - corr_original

        write_matrix_csv(segment_dir / "spearman_dU_dQ_original.csv", names, corr_original)
        write_matrix_csv(segment_dir / "spearman_dU_dQ_robust.csv", names, corr_robust)
        write_matrix_csv(segment_dir / "spearman_dU_dQ_difference.csv", names, difference)
        write_matrix_csv(segment_dir / "stevilo_parov_original.csv", names, count_original, integer=True)
        write_matrix_csv(segment_dir / "stevilo_parov_robust.csv", names, count_robust, integer=True)
        same = same_busbar_comparison(selected_metadata, corr_original, corr_robust, count_original, count_robust)
        same.write_csv(segment_dir / "korelacija_dU_dQ_iste_zbiralke_comparison.csv", separator=";", include_bom=True)
        write_public_metadata(segment_dir / "zbiralke_metadata.csv", selected_metadata)

        common_subtitle = (
            f"exact 15-minute pairs; segment: {segment.label}; "
            f"measurement-set mode: {args.measurement_set_mode}"
        )
        draw_heatmap(corr_original, names, segment_dir / "spearman_heatmap_original.png", segment_dir / "spearman_heatmap_original.svg", args.annotate_max_busbars, "Original Spearman correlation of ΔU and ΔQ", common_subtitle, "Spearman coefficient")
        draw_heatmap(corr_robust, names, segment_dir / "spearman_heatmap_robust.png", segment_dir / "spearman_heatmap_robust.svg", args.annotate_max_busbars, "Robust Spearman correlation of ΔU and ΔQ", common_subtitle + f"; validation: {args.pair_validation_mode}", "Spearman coefficient")
        draw_heatmap(difference, names, segment_dir / "spearman_heatmap_difference.png", segment_dir / "spearman_heatmap_difference.svg", args.annotate_max_busbars, "Correlation difference: robust - original", common_subtitle, "Δρ")

        finite_difference = difference[np.isfinite(difference)]
        summary = [
            "ROBUSTNA PRIMERJAVA SPEARMANOVE KORELACIJE dU-dQ 110/SN",
            f"Segment: {segment.label}",
            f"Pair validation mode: {args.pair_validation_mode}",
            f"Measurement set mode: {args.measurement_set_mode}",
            f"Minimalno skupnih parov: {args.min_pair_points}",
            f"Stevilo prikazanih lokacij (unija originalno/robustno veljavnih): {len(names)}",
            f"Veljavnih razlik koeficientov: {finite_difference.size}",
            f"Mediana |delta rho|: {np.median(np.abs(finite_difference)):.6f}" if finite_difference.size else "Mediana |delta rho|: ni veljavnih razlik",
            "Delta rho = rho_robust - rho_original.",
            "Q ostane v izvorni konvenciji: negativno kapacitivno, pozitivno induktivno.",
        ]
        (segment_dir / "izracun_povzetek.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
        print(f"Segment {segment.number}/{len(segments)} koncan: {segment.label}")

    pl.concat(all_validation, how="vertical_relaxed").write_csv(
        output_dir / "diagnostika_validacije_15min_parov.csv", separator=";", include_bom=True
    )
    pl.concat(all_warnings, how="vertical_relaxed").write_csv(
        output_dir / "opozorila_kakovosti.csv", separator=";", include_bom=True
    )
    pl.concat(all_appearance, how="vertical_relaxed").write_csv(
        output_dir / "diagnostika_pojavljanja_transformatorjev.csv", separator=";", include_bom=True
    )
    root_summary = [
        "ROBUSTNA ANALIZA dU-dQ 110/SN",
        f"Vhod: {input_path}",
        f"Strogih lokacij 110/SN: {loaded.metadata.height}",
        f"Strogih transformatorjev 110/SN: {loaded.metadata['n_expected_transformers'].sum()}",
        f"Segmentov: {len(segments)}",
        f"PAIR_VALIDATION_MODE={args.pair_validation_mode}",
        f"MEASUREMENT_SET_MODE={args.measurement_set_mode}",
        f"Sumljivih podvojenih zapisov: {loaded.duplicate_diagnostics.height}",
        "Originalna metoda: mediana U po veljavnem U naboru in vsota Q po veljavnem Q naboru.",
        "Robustna metoda: identiteta nabora se preveri med t in t+15; brez interpolacije.",
        "All_expected je namerno izbiren in ni privzet zaradi nevarnosti prevelike izgube podatkov.",
        "Predznak Q ni spremenjen.",
    ]
    (output_dir / "izracun_povzetek.txt").write_text("\n".join(root_summary) + "\n", encoding="utf-8")
    print("Izracun vseh zveznih segmentov je koncan.")
    print(f"Izhodna mapa: {output_dir}")


if __name__ == "__main__":
    main()
