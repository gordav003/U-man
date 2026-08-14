from __future__ import annotations

"""Korelacija napetosti in jalove moci iz dveh PowerFactory izvozov.

Glavni rezultat je Spearmanova korelacija tocnih 15-minutnih sprememb:
    dU(t) = U(t + 15 min) - U(t)
    dQ(t) = Q(t + 15 min) - Q(t)

Izracunana je tudi korelacija absolutnih nivojev kot dodatna diagnostika.
Vrednosti Q ostanejo v izvorni PowerFactory konvenciji Terminal i.
"""

import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr


BASE_VOLTAGE_KV = 110.0
DELTA_MINUTES = 15


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Korelacija PowerFactory U-Q izvoza.")
    parser.add_argument(
        "--voltage",
        type=Path,
        default=Path.home() / "Desktop" / "nap_PFD",
        help="PowerFactory izvoz napetosti v p.u.",
    )
    parser.add_argument(
        "--reactive-power",
        type=Path,
        default=Path.home() / "Desktop" / "Q_PFD",
        help="PowerFactory izvoz jalove moci v MVAr.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root() / "Uman meritve" / "korelacija_PowerFactory_nap_Q",
    )
    return parser.parse_args()


def read_powerfactory_export(path: Path) -> tuple[str, dict[datetime, float]]:
    if not path.is_file():
        raise FileNotFoundError(f"Datoteka ne obstaja: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        export_description = handle.readline().strip()
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or len(reader.fieldnames) != 2:
            raise ValueError(f"Nepricakovana glava izvoza v {path}")
        time_column, value_column = reader.fieldnames
        values: dict[datetime, float] = {}
        for line_number, row in enumerate(reader, start=3):
            timestamp = datetime.strptime(row[time_column].strip(), "%Y.%m.%d %H:%M:%S")
            if timestamp in values:
                raise ValueError(f"Podvojen timestamp {timestamp} v {path}, vrstica {line_number}")
            value = float(row[value_column])
            if not np.isfinite(value):
                raise ValueError(f"Neveljavna vrednost v {path}, vrstica {line_number}")
            values[timestamp] = value
    if len(values) < 3:
        raise ValueError(f"Premalo meritev v {path}")
    return export_description, values


def correlation(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan, np.nan, np.nan, np.nan
    spearman = spearmanr(x, y)
    pearson = pearsonr(x, y)
    return (
        float(spearman.statistic),
        float(spearman.pvalue),
        float(pearson.statistic),
        float(pearson.pvalue),
    )


def write_aligned_data(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    fields = [
        "time", "U_pu", "U_kV", "Q_MVAr", "dU_pu", "dU_kV", "dQ_MVAr"
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter=";")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def draw_scatter(
    output_dir: Path,
    u_kv: np.ndarray,
    q_mvar: np.ndarray,
    du_kv: np.ndarray,
    dq_mvar: np.ndarray,
    level_rho: float,
    change_rho: float,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), dpi=180)
    color = "#2463A8"
    axes[0].scatter(q_mvar, u_kv, s=17, alpha=0.65, color=color, edgecolors="none")
    axes[0].set_title(f"Absolute values: Spearman ρ = {level_rho:.3f}")
    axes[0].set_xlabel("Q / MVAr")
    axes[0].set_ylabel("U / kV")
    axes[0].grid(alpha=0.22)

    axes[1].scatter(dq_mvar, du_kv, s=17, alpha=0.65, color="#D97706", edgecolors="none")
    axes[1].axhline(0, color="#5B6470", linewidth=0.7)
    axes[1].axvline(0, color="#5B6470", linewidth=0.7)
    axes[1].set_title(f"Exact 15-minute changes: Spearman ρ = {change_rho:.3f}")
    axes[1].set_xlabel("ΔQ / MVAr")
    axes[1].set_ylabel("ΔU / kV")
    axes[1].grid(alpha=0.22)
    fig.suptitle("PowerFactory: voltage and reactive-power correlation", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "powerfactory_voltage_reactive_power_correlation.png", bbox_inches="tight")
    fig.savefig(output_dir / "powerfactory_voltage_reactive_power_correlation.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    voltage_path = args.voltage.resolve()
    q_path = args.reactive_power.resolve()
    output_dir = args.output_dir.resolve()
    voltage_description, voltage = read_powerfactory_export(voltage_path)
    q_description, reactive_power = read_powerfactory_export(q_path)

    voltage_times = set(voltage)
    q_times = set(reactive_power)
    common_times = sorted(voltage_times & q_times)
    if not common_times:
        raise ValueError("Izvoza nimata skupnih timestampov.")

    exact_pairs: list[tuple[datetime, datetime]] = []
    common_set = set(common_times)
    step = timedelta(minutes=DELTA_MINUTES)
    for timestamp in common_times:
        future = timestamp + step
        if future in common_set:
            exact_pairs.append((timestamp, future))

    u_pu = np.asarray([voltage[t] for t in common_times], dtype=float)
    u_kv = BASE_VOLTAGE_KV * u_pu
    q_mvar = np.asarray([reactive_power[t] for t in common_times], dtype=float)
    du_pu = np.asarray([voltage[f] - voltage[t] for t, f in exact_pairs], dtype=float)
    du_kv = BASE_VOLTAGE_KV * du_pu
    dq_mvar = np.asarray(
        [reactive_power[f] - reactive_power[t] for t, f in exact_pairs], dtype=float
    )

    level_stats = correlation(u_kv, q_mvar)
    change_stats = correlation(du_kv, dq_mvar)
    output_dir.mkdir(parents=True, exist_ok=True)

    changes_by_time = {
        timestamp: (du_pu[i], du_kv[i], dq_mvar[i])
        for i, (timestamp, _) in enumerate(exact_pairs)
    }
    aligned_rows: list[dict[str, object]] = []
    for i, timestamp in enumerate(common_times):
        changes = changes_by_time.get(timestamp)
        aligned_rows.append(
            {
                "time": timestamp.isoformat(sep=" "),
                "U_pu": f"{u_pu[i]:.9f}",
                "U_kV": f"{u_kv[i]:.9f}",
                "Q_MVAr": f"{q_mvar[i]:.9f}",
                "dU_pu": "" if changes is None else f"{changes[0]:.9f}",
                "dU_kV": "" if changes is None else f"{changes[1]:.9f}",
                "dQ_MVAr": "" if changes is None else f"{changes[2]:.9f}",
            }
        )
    write_aligned_data(output_dir / "poravnani_podatki_U_Q.csv", aligned_rows)

    with (output_dir / "rezultati_korelacije.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(
            ["analiza", "n_pairs", "spearman_rho", "spearman_p_value", "pearson_r", "pearson_p_value"]
        )
        writer.writerow(["absolutni_nivoji_U_Q", len(common_times), *[f"{v:.12g}" for v in level_stats]])
        writer.writerow(["tocne_15min_spremembe_dU_dQ", len(exact_pairs), *[f"{v:.12g}" for v in change_stats]])

    quality_rows = [
        ("voltage_export", voltage_description),
        ("reactive_power_export", q_description),
        ("n_voltage_rows", len(voltage)),
        ("n_q_rows", len(reactive_power)),
        ("n_common_timestamps", len(common_times)),
        ("n_voltage_only_timestamps", len(voltage_times - q_times)),
        ("n_q_only_timestamps", len(q_times - voltage_times)),
        ("n_exact_15min_pairs", len(exact_pairs)),
        ("start", common_times[0].isoformat(sep=" ")),
        ("end", common_times[-1].isoformat(sep=" ")),
        ("base_voltage_kV", BASE_VOLTAGE_KV),
        ("Q_sign_changed", False),
    ]
    with (output_dir / "diagnostika_podatkov.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["check", "value"])
        writer.writerows(quality_rows)

    draw_scatter(
        output_dir, u_kv, q_mvar, du_kv, dq_mvar, level_stats[0], change_stats[0]
    )
    summary = [
        "KORELACIJA POWERFACTORY U-Q",
        f"Obdobje: {common_times[0]} do {common_times[-1]}",
        f"Skupnih timestampov: {len(common_times)}",
        f"Tocnih 15-minutnih parov: {len(exact_pairs)}",
        f"Absolutni nivoji: Spearman rho={level_stats[0]:.6f}, p={level_stats[1]:.6g}",
        f"15-min spremembe: Spearman rho={change_stats[0]:.6f}, p={change_stats[1]:.6g}",
        f"15-min spremembe: Pearson r={change_stats[2]:.6f}, p={change_stats[3]:.6g}",
        "Napetost: iz p.u. pretvorjena v kV z osnovo 110 kV.",
        "Q: izvorna konvencija PowerFactory Terminal i; predznak ni obrnjen.",
        "Korelacija ne dokazuje vzrocnosti.",
    ]
    (output_dir / "povzetek.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))
    print(f"Izhodna mapa: {output_dir}")


if __name__ == "__main__":
    main()
