from __future__ import annotations

r"""
Korelacija 15-minutnih sprememb jalove moči in napetosti na istem transformatorju.

Analizirani model:
    dU_pu(t) = beta_0 + beta_1 * dQ_MVAr(t)

kjer velja:
    dQ_MVAr(t) = Q(t) - Q(t - 15 min)
    dU_pu(t)   = U_pu(t) - U_pu(t - 15 min)

Skripta:
  1. prebere eno TR_*.parquet datoteko,
  2. poišče stolpce time, Q in U,
  3. po želji uporabi filter kakovosti qst_no == 1,
  4. pretvori napetost v kV in p.u.,
  5. tvori samo pare, ki so časovno oddaljeni natanko 15 minut,
  6. meritve razdeli na zvezne segmente brez časovnih vrzeli,
  7. za vsak dovolj dolg segment ločeno izračuna Pearson r, Spearman r_s,
     R^2, beta_1 in beta_0,
  8. shrani CSV rezultate in SVG grafe po segmentih.

Meritve iz ločenih segmentov se nikoli ne združijo v isti koeficient.

Najlažji način uporabe:
  - spodaj spremeni INPUT_FILE,
  - nato zaženi:
        python -m correlations.reactive_power_voltage_delta_15min

Lahko pa datoteko podaš tudi v PowerShellu:
    python -m correlations.reactive_power_voltage_delta_15min `
      --input-file "C:\pot\do\TR_KRSKO_400_TR411.parquet"

POMEMBNO:
  Predznak beta_1 je odvisen od predznačne konvencije meritve Q. Skripta
  orientacije Q ne obrača samodejno.
"""

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import re
import sys

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


# =============================================================================
# NASTAVITVE, KI JIH NAJPOGOSTEJE SPREMENIŠ
# =============================================================================

INPUT_FILE = Path(
    r"C:\LEON\Projekti\2026\CRESYM-Uman\Uman meritve\Pridobljeno in urejeno\urejeno\Uman_parquet\component_files\TR_JESENICE_110_TR1.parquet"
)

# None pomeni celotno razpoložljivo obdobje.
START = "2025-10-18"  # primer: "2025-04-01"
END = "2025-10-18"    # primer: "2025-04-20"

# Če je None, se nominalna napetost poskusi prebrati iz imena datoteke.
NOMINAL_KV = None

# Privzeto predpostavimo, da je Q že v MVAr.
Q_VALUES_ARE_KVAR = False

# Če obstaja stolpec qst_no/quality/status, obdrži samo kakovost == 1.
QUALITY_FILTER = True

CHANGE_MINUTES = 15
MIN_POINTS = 10

# Po želji izloči nerealne skoke. None pomeni brez izločanja.
MAX_ABS_DQ_MVAR = None
MAX_ABS_DU_PU = None

OUTPUT_DIR = Path(__file__).resolve().parent / "korelacija_dQ_dU_15min"


# =============================================================================
# PODATKOVNA STRUKTURA REZULTATA
# =============================================================================

@dataclass(frozen=True)
class RegressionResult:
    n_points: int
    pearson_r: float
    spearman_r_s: float
    r_squared: float
    beta_1_pu_per_mvar: float
    beta_1_kv_per_mvar: float
    beta_0_pu: float
    beta_0_kv: float
    q_change_std_mvar: float
    u_change_std_pu: float


@dataclass(frozen=True)
class ContinuousSegment:
    number: int
    start: pd.Timestamp
    end: pd.Timestamp
    n_measurements: int

    @property
    def tag(self) -> str:
        return (
            f"segment_{self.number:03d}_"
            f"{self.start:%Y%m%d_%H%M}_do_{self.end:%Y%m%d_%H%M}"
        )


# =============================================================================
# ARGUMENTI UKAZNE VRSTICE
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Korelacija 15-minutnih sprememb Q in U na istem transformatorju."
        )
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=INPUT_FILE,
        help="TR_*.parquet datoteka. Če argument manjka, se uporabi INPUT_FILE v kodi.",
    )
    parser.add_argument(
        "--start",
        default=START,
        help="Začetek obdobja v ISO zapisu, npr. 2025-04-01.",
    )
    parser.add_argument(
        "--end",
        default=END,
        help="Vključni konec obdobja v ISO zapisu, npr. 2025-04-20.",
    )
    parser.add_argument(
        "--nominal-kv",
        type=float,
        default=NOMINAL_KV,
        help="Nominalna napetost strani transformatorja. Če manjka, se prebere iz imena.",
    )
    parser.add_argument(
        "--q-in-kvar",
        action="store_true",
        default=Q_VALUES_ARE_KVAR,
        help="Uporabi, če je Q v kVAr namesto MVAr.",
    )
    parser.add_argument(
        "--no-quality-filter",
        action="store_true",
        help="Izklopi filter kakovosti meritve.",
    )
    parser.add_argument(
        "--change-minutes",
        type=int,
        default=CHANGE_MINUTES,
        help="Interval spremembe v minutah. Privzeto: 15.",
    )
    parser.add_argument(
        "--max-abs-dq-mvar",
        type=float,
        default=MAX_ABS_DQ_MVAR,
        help="Po želji izloči pare z abs(dQ) nad podano vrednostjo MVAr.",
    )
    parser.add_argument(
        "--min-segment-points",
        type=int,
        default=MIN_POINTS,
        help=(
            "Najmanjše število parov dQ/dU za korelacijo posameznega "
            f"zveznega segmenta. Privzeto: {MIN_POINTS}."
        ),
    )
    parser.add_argument(
        "--max-abs-du-pu",
        type=float,
        default=MAX_ABS_DU_PU,
        help="Po želji izloči pare z abs(dU) nad podano vrednostjo p.u.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Izhodna mapa za CSV in SVG rezultate.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Grafe po shranjevanju tudi prikaži.",
    )
    return parser.parse_args()


# =============================================================================
# POMOŽNE FUNKCIJE
# =============================================================================

def safe_name(text: str) -> str:
    return re.sub(r"[^A-Z0-9-]+", "_", str(text).upper()).strip("_")


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    by_lower = {str(column).lower(): str(column) for column in df.columns}
    for candidate in candidates:
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    return None


def robust_to_numeric(values: pd.Series) -> pd.Series:
    """Pretvori številske ali tekstovne meritve v float.

    Podprte so decimalne vejice, nedeljivi presledki in morebitne enote,
    zapisane ob številu.
    """
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_numeric(values, errors="coerce")

    text = values.astype("string").str.strip()
    text = text.str.replace("\u00a0", "", regex=False)
    text = text.str.replace(" ", "", regex=False)
    text = text.str.replace(",", ".", regex=False)
    text = text.str.replace(r"[^0-9eE+\-.]", "", regex=True)
    return pd.to_numeric(text, errors="coerce")


def find_best_numeric_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> tuple[str | None, pd.Series | None, int]:
    """Med poimensko ustreznimi stolpci izbere tistega z največ številkami."""
    by_lower = {str(column).lower(): str(column) for column in df.columns}
    checked: list[tuple[int, int, str, pd.Series]] = []

    for priority, candidate in enumerate(candidates):
        column = by_lower.get(candidate.lower())
        if column is None:
            continue
        numeric = robust_to_numeric(df[column])
        checked.append((int(numeric.notna().sum()), -priority, column, numeric))

    if not checked:
        return None, None, 0

    valid_count, _, column, numeric = max(checked, key=lambda item: (item[0], item[1]))
    return column, numeric, valid_count


def parse_nominal_kv(path: Path, override: float | None) -> float:
    if override is not None:
        if override <= 0:
            raise ValueError("--nominal-kv mora biti večji od 0.")
        return float(override)

    voltage_tokens = [
        int(token)
        for token in path.stem.split("_")
        if token.isdigit() and 1 <= int(token) <= 1000
    ]

    if not voltage_tokens:
        raise ValueError(
            "Nominalne napetosti ni mogoče prebrati iz imena datoteke. "
            "Podaj jo z --nominal-kv, npr. --nominal-kv 110."
        )

    if len(voltage_tokens) > 1:
        print(
            "OPOZORILO: ime vsebuje več napetostnih nivojev "
            f"{voltage_tokens}; uporabljena bo prva vrednost {voltage_tokens[0]} kV. "
            "Po potrebi uporabi --nominal-kv."
        )

    return float(voltage_tokens[0])


def parse_optional_datetime(value: str | None, end_of_day: bool) -> pd.Timestamp | None:
    if value is None or str(value).strip() == "":
        return None

    timestamp = pd.Timestamp(value)

    # Če je podan samo datum in gre za konec obdobja, vključimo celoten dan.
    if end_of_day and len(str(value).strip()) <= 10:
        timestamp = timestamp + pd.Timedelta(days=1)

    return timestamp


def convert_voltage(
    values: pd.Series,
    nominal_kv: float,
) -> tuple[pd.Series, pd.Series, str]:
    numeric = pd.to_numeric(values, errors="coerce")
    median_abs = numeric.dropna().abs().median()

    if pd.isna(median_abs):
        raise ValueError("Napetostni stolpec nima veljavnih številskih vrednosti.")

    if 0.5 <= median_abs <= 1.5:
        u_pu = numeric
        u_kv = numeric * nominal_kv
        detected_unit = "p.u."
    elif median_abs > 1000:
        u_kv = numeric / 1000.0
        u_pu = u_kv / nominal_kv
        detected_unit = "V"
    else:
        u_kv = numeric
        u_pu = u_kv / nominal_kv
        detected_unit = "kV"

    return u_kv, u_pu, detected_unit


def read_transformer_data(
    path: Path,
    nominal_kv: float,
    quality_filter: bool,
    q_in_kvar: bool,
    start: pd.Timestamp | None,
    end_exclusive: pd.Timestamp | None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Datoteka ne obstaja: {path}\n"
            "Spremeni INPUT_FILE v kodi ali uporabi --input-file."
        )

    raw = pd.read_parquet(path)

    time_col = find_column(
        raw,
        ["time", "cas", "systime", "systime(UTC+1)", "period_start"],
    )
    q_col, q_numeric, q_valid_count = find_best_numeric_column(
        raw,
        ["Q", "q", "Q_MVAr", "q_mvar", "Q_kVAr", "q_kvar"],
    )
    u_col, u_numeric, u_valid_count = find_best_numeric_column(
        raw,
        ["U", "u", "U_kV", "u_kv", "U_pu", "u_pu", "voltage", "Voltage"],
    )
    quality_col = find_column(raw, ["qst_no", "qst_no_min", "quality", "status"])

    missing = []
    if time_col is None:
        missing.append("time")
    if q_col is None:
        missing.append("Q")
    if u_col is None:
        missing.append("U")
    if missing:
        raise RuntimeError(
            f"V {path.name} manjkajo zahtevani stolpci: {', '.join(missing)}. "
            f"Razpoložljivi stolpci: {list(raw.columns)}"
        )

    if q_valid_count == 0 or u_valid_count == 0:
        raise ValueError(
            f"Izbrana stolpca nimata uporabnih številskih vrednosti. "
            f"Q: {q_col!r} ({q_valid_count} veljavnih), "
            f"U: {u_col!r} ({u_valid_count} veljavnih).\n"
            f"Razpoložljivi stolpci: {list(raw.columns)}"
        )

    selected_columns = [time_col]
    if quality_col is not None:
        selected_columns.append(quality_col)
    selected_columns = list(dict.fromkeys(selected_columns))

    df = raw[selected_columns].copy()
    df = df.rename(columns={time_col: "time"})
    df["Q_raw"] = q_numeric
    df["U_raw"] = u_numeric
    if quality_col is not None and quality_col in df.columns:
        df = df.rename(columns={quality_col: "quality"})

    df["time"] = pd.to_datetime(df["time"], errors="coerce", dayfirst=True)

    n_raw = len(df)
    n_time_valid = int(df["time"].notna().sum())
    n_q_valid = int(df["Q_raw"].notna().sum())
    n_u_valid = int(df["U_raw"].notna().sum())
    df = df.dropna(subset=["time", "Q_raw", "U_raw"]).copy()

    n_before_quality = len(df)
    if df.empty:
        raise ValueError(
            "Ni niti ene vrstice, kjer so hkrati veljavni čas, Q in U.\n"
            f"Veljavni čas: {n_time_valid}/{n_raw}, "
            f"Q ({q_col}): {n_q_valid}/{n_raw}, "
            f"U ({u_col}): {n_u_valid}/{n_raw}.\n"
            "To običajno pomeni, da sta Q in U zapisana v različnih vrsticah "
            "ali da je bil izbran napačen stolpec."
        )

    quality_applied = False
    quality_filter_fallback = False
    quality_values = ""
    if quality_filter and "quality" in df.columns:
        quality_numeric = robust_to_numeric(df["quality"])
        unique_quality = sorted(quality_numeric.dropna().unique().tolist())
        quality_values = ";".join(str(value) for value in unique_quality[:20])
        filtered = df[quality_numeric == 1].copy()

        if filtered.empty:
            quality_filter_fallback = True
            print(
                "OPOZORILO: filter kakovosti == 1 bi odstranil vse veljavne "
                f"vrstice. Najdene vrednosti kakovosti: {unique_quality}. "
                "Analiza se nadaljuje brez filtra kakovosti."
            )
        else:
            df = filtered
            quality_applied = True

    if start is not None:
        df = df[df["time"] >= start].copy()
    if end_exclusive is not None:
        df = df[df["time"] < end_exclusive].copy()

    if df.empty:
        raise ValueError(
            "Po časovnem in kakovostnem filtriranju ni ostala nobena meritev. "
            "Preveri START/END ali izklopi filter kakovosti."
        )

    if q_in_kvar:
        df["Q_MVAr"] = df["Q_raw"] / 1000.0
        q_detected_unit = "kVAr (pretvorjeno v MVAr)"
    else:
        df["Q_MVAr"] = df["Q_raw"]
        q_detected_unit = "MVAr"

    df["U_kV"], df["U_pu"], u_detected_unit = convert_voltage(
        df["U_raw"],
        nominal_kv,
    )

    # Če je za isti čas več vrstic, uporabimo povprečje.
    before_duplicates = len(df)
    df = (
        df.groupby("time", as_index=False)[["Q_MVAr", "U_kV", "U_pu"]]
        .mean()
        .sort_values("time")
        .reset_index(drop=True)
    )
    duplicate_rows_aggregated = before_duplicates - len(df)

    metadata = {
        "input_file": str(path.resolve()),
        "file_name": path.name,
        "nominal_kv": nominal_kv,
        "detected_u_unit": u_detected_unit,
        "assumed_q_unit": q_detected_unit,
        "selected_q_column": q_col,
        "selected_u_column": u_col,
        "q_numeric_values_before_joint_filter": n_q_valid,
        "u_numeric_values_before_joint_filter": n_u_valid,
        "quality_column_found": quality_col is not None,
        "quality_filter_applied": quality_applied,
        "quality_filter_fallback_to_unfiltered": quality_filter_fallback,
        "quality_values_found": quality_values,
        "n_raw_rows": n_raw,
        "n_rows_after_numeric_filter": n_before_quality,
        "n_rows_after_all_filters": len(df),
        "duplicate_rows_aggregated": duplicate_rows_aggregated,
        "time_start": df["time"].min() if not df.empty else pd.NaT,
        "time_end": df["time"].max() if not df.empty else pd.NaT,
    }

    return df, metadata


def add_continuous_segments(
    df: pd.DataFrame,
    change_minutes: int,
) -> tuple[pd.DataFrame, list[ContinuousSegment]]:
    """Označi strogo zvezne segmente izbranega merilnega koraka.

    Nov segment se začne ob vsakem razmiku, ki ni natanko ``change_minutes``.
    Tako manjkajoča meritev, podvojeni/obrnjeni čas ali daljša vrzel ne morejo
    povezati dveh ločenih obdobij v isto korelacijo.
    """
    if change_minutes <= 0:
        raise ValueError("--change-minutes mora biti večji od 0.")

    expected = pd.Timedelta(minutes=change_minutes)
    result = label_continuous_segments(df, expected_interval=expected)

    segments = []
    for number, group in result.groupby("segment", sort=True):
        segments.append(
            ContinuousSegment(
                number=int(number),
                start=pd.Timestamp(group["time"].iloc[0]),
                end=pd.Timestamp(group["time"].iloc[-1]),
                n_measurements=len(group),
            )
        )
    return result, segments


def build_exact_change_pairs(
    df: pd.DataFrame,
    change_minutes: int,
) -> pd.DataFrame:
    if change_minutes <= 0:
        raise ValueError("--change-minutes mora biti večji od 0.")

    current = df.rename(
        columns={
            "Q_MVAr": "Q_MVAr_t",
            "U_kV": "U_kV_t",
            "U_pu": "U_pu_t",
        }
    )

    previous = df.rename(
        columns={
            "Q_MVAr": "Q_MVAr_prev",
            "U_kV": "U_kV_prev",
            "U_pu": "U_pu_prev",
        }
    ).copy()

    # Vrednost iz časa t-change_minutes prestavimo na čas t.
    previous["time"] = previous["time"] + pd.Timedelta(minutes=change_minutes)

    join_columns = ["time"]
    if "segment" in df.columns:
        join_columns.append("segment")
    paired = current.merge(
        previous,
        on=join_columns,
        how="inner",
        validate="one_to_one",
    )

    paired[f"dQ_{change_minutes}min_MVAr"] = (
        paired["Q_MVAr_t"] - paired["Q_MVAr_prev"]
    )
    paired[f"dU_{change_minutes}min_kV"] = (
        paired["U_kV_t"] - paired["U_kV_prev"]
    )
    paired[f"dU_{change_minutes}min_pu"] = (
        paired["U_pu_t"] - paired["U_pu_prev"]
    )

    paired["interval_start"] = paired["time"] - pd.Timedelta(minutes=change_minutes)
    paired["interval_end"] = paired["time"]

    return paired.sort_values("time").reset_index(drop=True)


def apply_change_filters(
    pairs: pd.DataFrame,
    change_minutes: int,
    max_abs_dq_mvar: float | None,
    max_abs_du_pu: float | None,
) -> tuple[pd.DataFrame, int]:
    dq_col = f"dQ_{change_minutes}min_MVAr"
    du_col = f"dU_{change_minutes}min_pu"

    mask = pairs[dq_col].notna() & pairs[du_col].notna()

    if max_abs_dq_mvar is not None:
        if max_abs_dq_mvar <= 0:
            raise ValueError("--max-abs-dq-mvar mora biti večji od 0.")
        mask &= pairs[dq_col].abs() <= max_abs_dq_mvar

    if max_abs_du_pu is not None:
        if max_abs_du_pu <= 0:
            raise ValueError("--max-abs-du-pu mora biti večji od 0.")
        mask &= pairs[du_col].abs() <= max_abs_du_pu

    filtered = pairs[mask].copy().reset_index(drop=True)
    return filtered, int((~mask).sum())


def regression_statistics(
    pairs: pd.DataFrame,
    change_minutes: int,
    nominal_kv: float,
    min_points: int = MIN_POINTS,
) -> RegressionResult:
    x_col = f"dQ_{change_minutes}min_MVAr"
    y_col = f"dU_{change_minutes}min_pu"

    valid = pairs[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    x = valid[x_col].astype(float)
    y = valid[y_col].astype(float)

    if min_points < 2:
        raise ValueError("--min-segment-points mora biti vsaj 2.")
    if len(valid) < min_points:
        raise RuntimeError(
            f"Premalo veljavnih {change_minutes}-minutnih parov: {len(valid)}. "
            f"Potrebnih je vsaj {min_points}."
        )

    if np.isclose(x.std(ddof=1), 0.0):
        raise RuntimeError("dQ nima variabilnosti, zato regresije ni mogoče izračunati.")
    if np.isclose(y.std(ddof=1), 0.0):
        raise RuntimeError("dU nima variabilnosti, zato regresije ni mogoče izračunati.")

    pearson_r = float(x.corr(y, method="pearson"))

    # Spearman brez odvisnosti od scipy: Pearsonova korelacija rangov.
    spearman_r = float(
        x.rank(method="average").corr(y.rank(method="average"), method="spearman")
    )

    beta_1, beta_0 = np.polyfit(x.to_numpy(), y.to_numpy(), deg=1)
    predicted = beta_0 + beta_1 * x.to_numpy()

    ss_res = float(np.sum((y.to_numpy() - predicted) ** 2))
    ss_tot = float(np.sum((y.to_numpy() - y.mean()) ** 2))
    r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    return RegressionResult(
        n_points=len(valid),
        pearson_r=pearson_r,
        spearman_r_s=spearman_r,
        r_squared=r_squared,
        beta_1_pu_per_mvar=float(beta_1),
        beta_1_kv_per_mvar=float(beta_1 * nominal_kv),
        beta_0_pu=float(beta_0),
        beta_0_kv=float(beta_0 * nominal_kv),
        q_change_std_mvar=float(x.std(ddof=1)),
        u_change_std_pu=float(y.std(ddof=1)),
    )


def apply_axis_style(axis: plt.Axes) -> None:
    axis.grid(True, which="major", linewidth=0.7, alpha=0.35)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(axis="both", labelsize=10)


def save_figure(figure: plt.Figure, path: Path, show: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, format="svg", bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


def plot_time_series(
    df: pd.DataFrame,
    output_path: Path,
    show: bool,
) -> None:
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(16, 8),
        sharex=True,
    )

    axes[0].plot(df["time"], df["Q_MVAr"], linewidth=1.15)
    axes[0].set_ylabel("Q / MVAr", fontsize=11)
    apply_axis_style(axes[0])

    axes[1].plot(df["time"], df["U_pu"], linewidth=1.15)
    axes[1].set_ylabel("U / p.u.", fontsize=11)
    axes[1].set_xlabel("Time", fontsize=11)
    apply_axis_style(axes[1])

    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    axes[1].xaxis.set_major_locator(locator)
    axes[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    figure.subplots_adjust(left=0.08, right=0.99, top=0.98, bottom=0.10, hspace=0.10)
    save_figure(figure, output_path, show)


def plot_scatter_regression(
    pairs: pd.DataFrame,
    result: RegressionResult,
    change_minutes: int,
    output_path: Path,
    show: bool,
) -> None:
    x_col = f"dQ_{change_minutes}min_MVAr"
    y_col = f"dU_{change_minutes}min_kV"

    x = pairs[x_col].to_numpy(dtype=float)
    y = pairs[y_col].to_numpy(dtype=float)

    figure, axis = plt.subplots(figsize=(9.5, 7.5))
    axis.scatter(x, y, s=18, alpha=0.45, edgecolors="none")

    x_line = np.linspace(np.nanmin(x), np.nanmax(x), 250)
    y_line = result.beta_0_kv + result.beta_1_kv_per_mvar * x_line
    axis.plot(x_line, y_line, linewidth=1.8, label="Linear regression")

    axis.axhline(0.0, linewidth=0.9, alpha=0.55)
    axis.axvline(0.0, linewidth=0.9, alpha=0.55)
    axis.set_xlabel("dQ/MVAr", fontsize=16, fontweight="bold", labelpad=10)
    axis.set_ylabel("dU/kV", fontsize=16, fontweight="bold", labelpad=10)
    apply_axis_style(axis)

    annotation = (
        f"Pearson r = {result.pearson_r:.3f}\n"
        f"Spearman rₛ = {result.spearman_r_s:.3f}\n"
        f"R² = {result.r_squared:.3f}\n"
        f"β₁ = {result.beta_1_kv_per_mvar:.4f} kV/MVAr\n"
        f"n = {result.n_points}"
    )
    axis.text(
        0.03,
        0.97,
        annotation,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.9},
    )
    axis.legend(loc="lower right", frameon=False)

    figure.tight_layout()
    save_figure(figure, output_path, show)


def correlation_strength(value: float) -> str:
    absolute = abs(value)
    if absolute < 0.10:
        return "praktično ni linearne povezave"
    if absolute < 0.30:
        return "šibka povezava"
    if absolute < 0.50:
        return "zmerna povezava"
    if absolute < 0.70:
        return "zmerno močna povezava"
    if absolute < 0.90:
        return "močna povezava"
    return "zelo močna povezava"


def direction_text(value: float) -> str:
    if value > 0:
        return "pozitivna"
    if value < 0:
        return "negativna"
    return "brez smeri"


# =============================================================================
# GLAVNI PROGRAM
# =============================================================================

def main() -> None:
    args = parse_args()

    input_file = args.input_file.resolve()
    output_root = args.output_dir.resolve()
    nominal_kv = parse_nominal_kv(input_file, args.nominal_kv)
    start = parse_optional_datetime(args.start, end_of_day=False)
    end_exclusive = parse_optional_datetime(args.end, end_of_day=True)

    if start is not None and end_exclusive is not None and start >= end_exclusive:
        raise ValueError("Začetek obdobja mora biti pred koncem obdobja.")

    quality_filter = QUALITY_FILTER and not args.no_quality_filter

    data, metadata = read_transformer_data(
        path=input_file,
        nominal_kv=nominal_kv,
        quality_filter=quality_filter,
        q_in_kvar=args.q_in_kvar,
        start=start,
        end_exclusive=end_exclusive,
    )

    if data.empty:
        raise RuntimeError("Po filtriranju ni ostala nobena veljavna meritev.")

    analysis_dir = output_root / safe_name(input_file.stem)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    data, segments = add_continuous_segments(data, args.change_minutes)
    all_pairs = build_exact_change_pairs(data, args.change_minutes)
    data.to_csv(
        analysis_dir / "01_original_timeseries_Q_U.csv",
        index=False,
        encoding="utf-8-sig",
    )
    all_pairs.to_csv(
        analysis_dir / f"02_exact_{args.change_minutes}min_changes_all_segments.csv",
        index=False,
        encoding="utf-8-sig",
    )
    plot_time_series(data, analysis_dir / "01_Q_U_time_series.svg", args.show)

    segment_rows: list[dict[str, object]] = []
    valid_results: list[tuple[ContinuousSegment, RegressionResult]] = []
    total_filtered_changes = 0

    for segment in segments:
        segment_data = data[data["segment"] == segment.number].copy()
        segment_all_pairs = all_pairs[all_pairs["segment"] == segment.number].copy()
        segment_pairs, n_filtered_changes = apply_change_filters(
            segment_all_pairs,
            args.change_minutes,
            args.max_abs_dq_mvar,
            args.max_abs_du_pu,
        )
        total_filtered_changes += n_filtered_changes

        row: dict[str, object] = {
            "segment": segment.number,
            "segment_tag": segment.tag,
            "start": segment.start,
            "end": segment.end,
            "n_measurements": segment.n_measurements,
            "n_exact_pairs_before_change_filter": len(segment_all_pairs),
            "n_pairs_removed_by_change_filter": n_filtered_changes,
            "n_used_pairs": len(segment_pairs),
            "status": "preskočen",
            "reason": "",
        }

        try:
            result = regression_statistics(
                segment_pairs,
                args.change_minutes,
                nominal_kv,
                min_points=args.min_segment_points,
            )
        except RuntimeError as error:
            row["reason"] = str(error)
            segment_rows.append(row)
            continue

        row.update(asdict(result))
        row["status"] = "izračunan"
        segment_rows.append(row)
        valid_results.append((segment, result))

        segment_dir = analysis_dir / segment.tag
        segment_dir.mkdir(parents=True, exist_ok=True)
        segment_data.to_csv(
            segment_dir / "01_measurements_Q_U.csv",
            index=False,
            encoding="utf-8-sig",
        )
        segment_pairs.to_csv(
            segment_dir / f"02_exact_{args.change_minutes}min_changes.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame([row]).to_csv(
            segment_dir / "03_correlation_regression_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
        plot_time_series(
            segment_data,
            segment_dir / "01_Q_U_time_series.svg",
            args.show,
        )
        plot_scatter_regression(
            segment_pairs,
            result,
            args.change_minutes,
            segment_dir / f"02_scatter_dQ_dU_{args.change_minutes}min.svg",
            args.show,
        )

    pd.DataFrame(segment_rows).to_csv(
        analysis_dir / "03_correlation_regression_by_segment.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metadata.update(
        {
            "change_minutes": args.change_minutes,
            "min_segment_points": args.min_segment_points,
            "n_continuous_segments": len(segments),
            "n_segments_with_result": len(valid_results),
            "n_exact_change_pairs_all_segments": len(all_pairs),
            "n_pairs_removed_by_change_filter": total_filtered_changes,
            "output_dir": str(analysis_dir),
        }
    )
    pd.DataFrame([metadata]).to_csv(
        analysis_dir / "00_metadata.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if not valid_results:
        raise RuntimeError(
            "Noben zvezni segment nima dovolj uporabnih in spremenljivih "
            "parov za korelacijo. Podrobnosti so v indeksu segmentov."
        )

    print("=" * 100)
    print(f"KORELACIJA {args.change_minutes}-MINUTNIH SPREMEMB Q IN U NA TRANSFORMATORJU")
    print("=" * 100)
    print(f"Datoteka:                {input_file.name}")
    print(f"Nominalna napetost:      {nominal_kv:g} kV")
    print(f"Zaznana enota U:         {metadata['detected_u_unit']}")
    print(f"Predpostavljena enota Q: {metadata['assumed_q_unit']}")
    print(f"Časovno obdobje:         {metadata['time_start']} do {metadata['time_end']}")
    print(f"Veljavnih meritev:       {len(data)}")
    print(f"Zveznih segmentov:       {len(segments)}")
    print(f"Segmentov z rezultatom:  {len(valid_results)}")
    print(f"Natančnih {args.change_minutes}-min parov: {len(all_pairs)}")
    print("-" * 100)
    for segment, result in valid_results:
        print(
            f"{segment.tag}: n={result.n_points}, "
            f"Pearson r={result.pearson_r:.5f}, "
            f"Spearman r_s={result.spearman_r_s:.5f}, "
            f"R^2={result.r_squared:.5f}, "
            f"beta_1={result.beta_1_kv_per_mvar:.6f} kV/MVAr"
        )
    print(
        "POMEMBNO: predznak beta_1 je odvisen od predznačne konvencije Q "
        "in orientacije meritve na transformatorju."
    )
    print("-" * 100)
    print(f"Rezultati: {analysis_dir}")
    print("=" * 100)


if __name__ == "__main__":
    main()
