from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd


APP_TITLE = "Kompenzacija jalove moči RTP"
COSPHI_LIMIT_DEFAULT = 0.95
HV_VOLTAGE_KV = 110
MV_VOLTAGE_KV_CANDIDATES = (10, 20, 35)
MIN_POINTS = 10
QUALITY_FILTER = True


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_data_directory() -> Path:
    return (
        project_root()
        / "Uman meritve"
        / "Pridobljeno in urejeno"
        / "urejeno"
        / "Uman_parquet"
        / "component_files"
    )


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^\w\-.]+", "_", str(value), flags=re.UNICODE)
    return re.sub(r"_+", "_", cleaned).strip("_")


def find_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    by_lower = {str(column).lower(): str(column) for column in columns}
    for candidate in candidates:
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    return None


@dataclass(frozen=True)
class TransformerFile:
    path: Path
    rtp: str
    transformer_id: str
    voltage_kv: int


def parse_transformer_file(path: Path) -> TransformerFile | None:
    """Razčleni standardno ime TR_<RTP>_<kV>_<transformator>.parquet."""
    parts = path.stem.split("_")
    if len(parts) < 4 or parts[0].upper() != "TR":
        return None

    voltage_indices = [index for index, part in enumerate(parts) if part.isdigit()]
    if len(voltage_indices) != 1:
        return None

    voltage_index = voltage_indices[0]
    rtp = "_".join(parts[1:voltage_index])
    transformer_id = "_".join(parts[voltage_index + 1 :])
    if not rtp or not transformer_id:
        return None

    return TransformerFile(
        path=path,
        rtp=rtp,
        transformer_id=transformer_id,
        voltage_kv=int(parts[voltage_index]),
    )


def discover_rtp_files(data_dir: Path) -> tuple[dict[str, list[Path]], dict[str, int]]:
    """Poišče iste stroge pare 110/SN kot Reactive_Power_Analysis.py."""
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Mapa ne obstaja: {data_dir}")

    parquet_files = sorted(data_dir.glob("*.parquet"))
    records = [
        record
        for path in parquet_files
        if (record := parse_transformer_file(path)) is not None
    ]

    by_key: dict[tuple[str, str], dict[int, list[TransformerFile]]] = {}
    for record in records:
        by_key.setdefault((record.rtp, record.transformer_id), {}).setdefault(
            record.voltage_kv, []
        ).append(record)

    rtp_files: dict[str, list[Path]] = {}
    excluded_without_mv = 0
    excluded_with_higher_voltage = 0

    for (rtp, _transformer_id), by_voltage in sorted(by_key.items()):
        hv_records = by_voltage.get(HV_VOLTAGE_KV, [])
        if not hv_records:
            continue
        if any(voltage > HV_VOLTAGE_KV for voltage in by_voltage):
            excluded_with_higher_voltage += len(hv_records)
            continue
        if not any(voltage in by_voltage for voltage in MV_VOLTAGE_KV_CANDIDATES):
            excluded_without_mv += len(hv_records)
            continue
        rtp_files.setdefault(rtp, []).extend(record.path for record in hv_records)

    for paths in rtp_files.values():
        paths.sort(key=lambda path: path.name.casefold())

    stats = {
        "all_parquet": len(parquet_files),
        "transformer_files": len(records),
        "rtp_count": len(rtp_files),
        "hv_files": sum(len(paths) for paths in rtp_files.values()),
        "excluded_without_mv": excluded_without_mv,
        "excluded_with_higher_voltage": excluded_with_higher_voltage,
    }
    return dict(sorted(rtp_files.items())), stats


def read_transformer_power(path: Path) -> pd.DataFrame:
    """Prebere in očisti meritve P in Q na 110-kV strani transformatorja."""
    raw = pd.read_parquet(path)
    time_column = find_column(
        raw.columns, ("time", "cas", "systime", "systime(UTC+1)", "period_start")
    )
    p_column = find_column(raw.columns, ("P", "P_MW"))
    q_column = find_column(raw.columns, ("Q", "Q_MVAr"))
    quality_column = find_column(
        raw.columns, ("qst_no", "qst_no_min", "quality", "status")
    )
    if time_column is None or p_column is None or q_column is None:
        raise RuntimeError(f"Datoteka {path.name} nima stolpcev time, P in Q.")

    selected_columns = list(
        dict.fromkeys(
            [
                time_column,
                p_column,
                q_column,
                *([quality_column] if quality_column is not None else []),
            ]
        )
    )
    frame = raw[selected_columns].copy().rename(
        columns={time_column: "time", p_column: "P_MW", q_column: "Q_MVAr"}
    )
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce", dayfirst=True)
    for column in ("P_MW", "Q_MVAr"):
        if frame[column].dtype == "object":
            frame[column] = frame[column].astype(str).str.replace(",", ".", regex=False)
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if QUALITY_FILTER and quality_column is not None:
        frame[quality_column] = pd.to_numeric(frame[quality_column], errors="coerce")
        frame = frame[frame[quality_column] == 1]

    frame = frame.dropna(subset=["time", "P_MW", "Q_MVAr"])
    return (
        frame.groupby("time", as_index=False)[["P_MW", "Q_MVAr"]]
        .mean()
        .sort_values("time")
        .reset_index(drop=True)
    )


def load_rtp_power(paths: Iterable[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    for path in paths:
        try:
            frame = read_transformer_power(path)
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")
            continue
        if not frame.empty:
            frame = frame.assign(transformer=path.stem)
            frames.append(frame)

    if not frames:
        details = "\n".join(errors[:5])
        raise RuntimeError(f"Za izbrani RTP ni veljavnih meritev.\n{details}")

    combined = pd.concat(frames, ignore_index=True)
    result = (
        combined.groupby("time", as_index=False)
        .agg(
            P_MW_sum=("P_MW", "sum"),
            Q_MVAr_sum=("Q_MVAr", "sum"),
            n_valid_transformers=("transformer", "nunique"),
        )
        .sort_values("time")
        .reset_index(drop=True)
    )
    if len(result) < MIN_POINTS:
        raise RuntimeError(
            f"Premalo veljavnih meritev: {len(result)} (zahtevanih vsaj {MIN_POINTS})."
        )
    return result


def calculate_compensation(
    source: pd.DataFrame,
    *,
    capacitive_mvar: float,
    inductive_mvar: float,
    cosphi_limit: float,
    use_abs_p: bool,
) -> pd.DataFrame:
    """Izračuna novi Q in presežek glede na dovoljeno območje cos(phi)."""
    if not 0 < cosphi_limit < 1:
        raise ValueError("Meja cos(phi) mora biti med 0 in 1.")
    if capacitive_mvar < 0 or inductive_mvar < 0:
        raise ValueError("Kapacitivna in induktivna moč ne smeta biti negativni.")

    frame = source.copy()
    frame["Q_original_MVAr"] = frame["Q_MVAr_sum"]
    frame["Q_compensated_MVAr"] = (
        frame["Q_original_MVAr"] + inductive_mvar - capacitive_mvar
    )
    frame["Q_MVAr_sum"] = frame["Q_compensated_MVAr"]

    factor = np.sqrt(1 - cosphi_limit**2) / cosphi_limit
    p_reference = frame["P_MW_sum"].abs() if use_abs_p else frame["P_MW_sum"]
    frame["Q_limit_ind_MVAr"] = p_reference * factor
    frame["Q_limit_cap_MVAr"] = -p_reference * factor

    q = frame["Q_MVAr_sum"]
    d_q = pd.Series(0.0, index=frame.index)
    status = pd.Series("V mejah", index=frame.index, dtype="object")
    capacitive_mask = q < frame["Q_limit_cap_MVAr"]
    inductive_mask = q > frame["Q_limit_ind_MVAr"]

    # Vrstni red je namenoma enak kot v Reactive_Power_Analysis.py.
    d_q.loc[capacitive_mask] = (
        q.loc[capacitive_mask] - frame.loc[capacitive_mask, "Q_limit_cap_MVAr"]
    )
    status.loc[capacitive_mask] = "Prekomerna kapacitivna oddaja"
    d_q.loc[inductive_mask] = (
        q.loc[inductive_mask] - frame.loc[inductive_mask, "Q_limit_ind_MVAr"]
    )
    status.loc[inductive_mask] = "Prekomeren induktivni odjem"

    frame["dQ_MVAr"] = d_q
    frame["status_q"] = status
    return frame


def scenario_statistics(frame: pd.DataFrame) -> dict[str, float | int]:
    count = len(frame)
    n_cap = int((frame["dQ_MVAr"] < 0).sum())
    n_ind = int((frame["dQ_MVAr"] > 0).sum())
    n_ok = count - n_cap - n_ind
    return {
        "count": count,
        "n_cap": n_cap,
        "n_ok": n_ok,
        "n_ind": n_ind,
        "pct_cap": 100.0 * n_cap / count,
        "pct_ok": 100.0 * n_ok / count,
        "pct_ind": 100.0 * n_ind / count,
        "dq_min": float(frame["dQ_MVAr"].min()),
        "dq_max": float(frame["dQ_MVAr"].max()),
    }


def duration_curve(frame: pd.DataFrame) -> tuple[np.ndarray, pd.Series]:
    sorted_dq = frame["dQ_MVAr"].sort_values().reset_index(drop=True)
    if len(sorted_dq) <= 1:
        percentage = np.zeros(len(sorted_dq))
    else:
        percentage = np.arange(len(sorted_dq)) / (len(sorted_dq) - 1) * 100.0
    return percentage, sorted_dq


def build_figure(
    rtp: str,
    original: pd.DataFrame,
    compensated: pd.DataFrame,
    *,
    capacitive_mvar: float,
    inductive_mvar: float,
    cosphi_limit: float,
):
    from matplotlib.figure import Figure

    figure = Figure(figsize=(10, 7), dpi=100)
    axis = figure.add_subplot(111)
    original_x, original_dq = duration_curve(original)
    x, dq = duration_curve(compensated)

    if capacitive_mvar != 0 or inductive_mvar != 0:
        axis.plot(
            original_x,
            original_dq,
            color="#777777",
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
            label="Original state",
        )

    masks = (
        (dq < 0, "Overcompensated (cap)", "#f2c14e"),
        (dq == 0, "Within limits", "#ff7f0e"),
        (dq > 0, "Undercompensated (ind)", "#1f77b4"),
    )
    for mask, label, color in masks:
        if mask.any():
            axis.plot(x[mask], dq[mask], linewidth=1.5, label=label, color=color)

    cap_mask = dq < 0
    ind_mask = dq > 0
    if cap_mask.any():
        axis.axvline(x[cap_mask].max(), color="red", alpha=0.45, linewidth=1.0)
    if ind_mask.any():
        axis.axvline(x[ind_mask].min(), color="red", alpha=0.45, linewidth=1.0)
    axis.axhline(0, color="#333333", linewidth=0.8, alpha=0.55)
    axis.grid(True, alpha=0.3)
    axis.set_xlim(0, 100)
    axis.set_xlabel("Percentage of measurement period / %")
    axis.set_ylabel("ΔQ / MVAr")
    title = f"Excess reactive-power consumption/generation - RTP {rtp}"
    if capacitive_mvar != 0 or inductive_mvar != 0:
        title += " (compensated)"
    axis.set_title(title)
    axis.legend(loc="best")
    figure.tight_layout()
    return figure


class CompensationApp:
    def __init__(self, root, initial_data_dir: Path):
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1280x820")
        self.root.minsize(980, 650)

        self.rtp_files: dict[str, list[Path]] = {}
        self.cache: dict[str, pd.DataFrame] = {}
        self.current_original: pd.DataFrame | None = None
        self.current_result: pd.DataFrame | None = None
        self.figure = None
        self.canvas = None
        self.toolbar = None

        self.data_dir_var = tk.StringVar(value=str(initial_data_dir))
        self.rtp_var = tk.StringVar()
        self.capacitive_var = tk.StringVar(value="0")
        self.inductive_var = tk.StringVar(value="0")
        self.cosphi_var = tk.StringVar(value=f"{COSPHI_LIMIT_DEFAULT:g}")
        self.use_abs_p_var = tk.BooleanVar(value=False)
        self.net_var = tk.StringVar(value="Neto sprememba Q: +0,000 MVAr")
        self.summary_var = tk.StringVar(value="Izberi mapo in RTP.")
        self.status_var = tk.StringVar(value="Pripravljeno.")

        self._build_ui()
        self.scan_directory(show_errors=False)

    def _build_ui(self) -> None:
        from tkinter import ttk

        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=self.tk.BOTH, expand=True)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        controls = ttk.Frame(main)
        controls.grid(row=0, column=0, sticky="nsw", padx=(0, 10))
        controls.columnconfigure(0, weight=1)

        source = ttk.LabelFrame(controls, text="1. Podatki", padding=8)
        source.grid(row=0, column=0, sticky="ew")
        source.columnconfigure(0, weight=1)
        ttk.Entry(source, textvariable=self.data_dir_var, width=44).grid(
            row=0, column=0, columnspan=2, sticky="ew"
        )
        ttk.Button(source, text="Izberi mapo ...", command=self.choose_directory).grid(
            row=1, column=0, sticky="ew", pady=(6, 0), padx=(0, 4)
        )
        ttk.Button(source, text="Osveži seznam", command=self.scan_directory).grid(
            row=1, column=1, sticky="ew", pady=(6, 0)
        )

        selection = ttk.LabelFrame(controls, text="2. Izbira RTP", padding=8)
        selection.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        selection.columnconfigure(0, weight=1)
        self.rtp_combo = ttk.Combobox(
            selection, textvariable=self.rtp_var, state="readonly", width=38
        )
        self.rtp_combo.grid(row=0, column=0, sticky="ew")
        self.rtp_combo.bind("<<ComboboxSelected>>", lambda _event: self.recalculate())

        compensation = ttk.LabelFrame(
            controls, text="3. Dodana jalova moč", padding=8
        )
        compensation.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        compensation.columnconfigure(1, weight=1)
        ttk.Label(compensation, text="Kapacitivna [MVAr]:").grid(
            row=0, column=0, sticky="w"
        )
        cap_entry = ttk.Entry(
            compensation, textvariable=self.capacitive_var, width=14
        )
        cap_entry.grid(row=0, column=1, sticky="ew")
        ttk.Label(compensation, text="Induktivna [MVAr]:").grid(
            row=1, column=0, sticky="w", pady=(6, 0)
        )
        ind_entry = ttk.Entry(
            compensation, textvariable=self.inductive_var, width=14
        )
        ind_entry.grid(row=1, column=1, sticky="ew", pady=(6, 0))
        ttk.Label(
            compensation,
            text="Q_novi = Q_izmerjeni + Q_ind − Q_cap",
            foreground="#555555",
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(7, 0))
        ttk.Label(compensation, textvariable=self.net_var).grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(3, 0)
        )

        settings = ttk.LabelFrame(controls, text="4. Nastavitve", padding=8)
        settings.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        settings.columnconfigure(1, weight=1)
        ttk.Label(settings, text="Meja cos(φ):").grid(row=0, column=0, sticky="w")
        cosphi_entry = ttk.Entry(settings, textvariable=self.cosphi_var, width=12)
        cosphi_entry.grid(row=0, column=1, sticky="ew")
        ttk.Checkbutton(
            settings,
            text="Za meji uporabi |P|",
            variable=self.use_abs_p_var,
            command=self.recalculate,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))

        buttons = ttk.Frame(controls)
        buttons.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        buttons.columnconfigure((0, 1), weight=1)
        ttk.Button(buttons, text="Izračunaj in izriši", command=self.recalculate).grid(
            row=0, column=0, columnspan=2, sticky="ew"
        )
        ttk.Button(buttons, text="Ponastavi", command=self.reset_compensation).grid(
            row=1, column=0, sticky="ew", pady=(6, 0), padx=(0, 4)
        )
        ttk.Button(buttons, text="Shrani graf ...", command=self.save_figure).grid(
            row=1, column=1, sticky="ew", pady=(6, 0)
        )
        ttk.Button(buttons, text="Izvozi rezultate CSV ...", command=self.export_csv).grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0)
        )

        summary = ttk.LabelFrame(controls, text="Rezultat", padding=8)
        summary.grid(row=5, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(
            summary,
            textvariable=self.summary_var,
            justify="left",
            font=("TkDefaultFont", 9),
        ).pack(anchor="w")

        plot_frame = ttk.LabelFrame(main, text="Graf", padding=5)
        plot_frame.grid(row=0, column=1, sticky="nsew")
        plot_frame.rowconfigure(0, weight=1)
        plot_frame.columnconfigure(0, weight=1)
        self.plot_frame = plot_frame

        ttk.Label(
            main,
            textvariable=self.status_var,
            relief=self.tk.SUNKEN,
            anchor="w",
            padding=4,
        ).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        for entry in (cap_entry, ind_entry, cosphi_entry):
            entry.bind("<Return>", lambda _event: self.recalculate())

    @staticmethod
    def _parse_number(value: str, label: str) -> float:
        try:
            return float(value.strip().replace(",", "."))
        except ValueError as exc:
            raise ValueError(f"Polje '{label}' mora vsebovati število.") from exc

    def choose_directory(self) -> None:
        from tkinter import filedialog

        selected = filedialog.askdirectory(
            title="Izberi mapo component_files",
            initialdir=self.data_dir_var.get() or str(project_root()),
        )
        if selected:
            self.data_dir_var.set(selected)
            self.scan_directory()

    def scan_directory(self, show_errors: bool = True) -> None:
        from tkinter import messagebox

        try:
            data_dir = Path(self.data_dir_var.get()).expanduser().resolve()
            self.rtp_files, stats = discover_rtp_files(data_dir)
            if not self.rtp_files:
                raise RuntimeError("V mapi ni veljavnih RTP s pari transformatorjev 110/SN.")
        except Exception as exc:
            self.rtp_files = {}
            self.rtp_combo["values"] = []
            self.rtp_var.set("")
            self.status_var.set(str(exc))
            if show_errors:
                messagebox.showerror("Napaka pri iskanju podatkov", str(exc))
            return

        self.cache.clear()
        rtps = list(self.rtp_files)
        self.rtp_combo["values"] = rtps
        if self.rtp_var.get() not in self.rtp_files:
            self.rtp_var.set(rtps[0])
        self.status_var.set(
            f"Najdenih {stats['rtp_count']} RTP in {stats['hv_files']} veljavnih "
            "110-kV transformatorskih datotek."
        )
        self.recalculate()

    def _scenario_inputs(self) -> tuple[float, float, float]:
        capacitive = self._parse_number(self.capacitive_var.get(), "Kapacitivna")
        inductive = self._parse_number(self.inductive_var.get(), "Induktivna")
        cosphi = self._parse_number(self.cosphi_var.get(), "Meja cos(phi)")
        return capacitive, inductive, cosphi

    def recalculate(self) -> None:
        from tkinter import messagebox

        rtp = self.rtp_var.get()
        if not rtp or rtp not in self.rtp_files:
            return
        try:
            capacitive, inductive, cosphi = self._scenario_inputs()
            self.status_var.set(f"Računam RTP {rtp} ...")
            self.root.update_idletasks()
            if rtp not in self.cache:
                self.cache[rtp] = load_rtp_power(self.rtp_files[rtp])
            source = self.cache[rtp]
            original = calculate_compensation(
                source,
                capacitive_mvar=0,
                inductive_mvar=0,
                cosphi_limit=cosphi,
                use_abs_p=self.use_abs_p_var.get(),
            )
            result = calculate_compensation(
                source,
                capacitive_mvar=capacitive,
                inductive_mvar=inductive,
                cosphi_limit=cosphi,
                use_abs_p=self.use_abs_p_var.get(),
            )
        except Exception as exc:
            self.status_var.set(str(exc))
            messagebox.showerror("Napaka pri izračunu", str(exc))
            return

        self.current_original = original
        self.current_result = result
        net = inductive - capacitive
        self.net_var.set(f"Neto sprememba Q: {net:+.3f} MVAr")
        original_statistics = scenario_statistics(original)
        statistics = scenario_statistics(result)
        self.summary_var.set(
            f"Merilnih točk: {statistics['count']:,}\n"
            "Izvorno → kompenzirano:\n"
            f"Kapacitivno: {original_statistics['pct_cap']:.2f} → "
            f"{statistics['pct_cap']:.2f} %\n"
            f"V mejah: {original_statistics['pct_ok']:.2f} → "
            f"{statistics['pct_ok']:.2f} %\n"
            f"Induktivno: {original_statistics['pct_ind']:.2f} → "
            f"{statistics['pct_ind']:.2f} %\n"
            f"Kap. vrh: {original_statistics['dq_min']:.3f} → "
            f"{statistics['dq_min']:.3f} MVAr\n"
            f"Ind. vrh: {original_statistics['dq_max']:.3f} → "
            f"{statistics['dq_max']:.3f} MVAr"
        )
        figure = build_figure(
            rtp,
            original,
            result,
            capacitive_mvar=capacitive,
            inductive_mvar=inductive,
            cosphi_limit=cosphi,
        )
        self._show_figure(figure)
        self.status_var.set(
            f"Izrisan RTP {rtp}; transformatorjev: {len(self.rtp_files[rtp])}."
        )

    def _show_figure(self, figure) -> None:
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

        if self.toolbar is not None:
            self.toolbar.destroy()
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
        self.figure = figure
        self.canvas = FigureCanvasTkAgg(figure, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.toolbar = NavigationToolbar2Tk(
            self.canvas, self.plot_frame, pack_toolbar=False
        )
        self.toolbar.update()
        self.toolbar.grid(row=1, column=0, sticky="ew")

    def reset_compensation(self) -> None:
        self.capacitive_var.set("0")
        self.inductive_var.set("0")
        self.recalculate()

    def save_figure(self) -> None:
        from tkinter import filedialog, messagebox

        if self.figure is None:
            messagebox.showwarning("Ni grafa", "Najprej izriši graf.")
            return
        rtp = safe_name(self.rtp_var.get())
        selected = filedialog.asksaveasfilename(
            title="Shrani graf kompenzacije",
            initialfile=f"{rtp}_kompenzacija_jalove_moci.svg",
            defaultextension=".svg",
            filetypes=[
                ("SVG vektorska slika", "*.svg"),
                ("PNG slika", "*.png"),
                ("PDF dokument", "*.pdf"),
            ],
        )
        if selected:
            try:
                self.figure.savefig(selected, dpi=180, bbox_inches="tight")
                self.status_var.set(f"Graf shranjen: {selected}")
            except Exception as exc:
                messagebox.showerror("Napaka pri shranjevanju", str(exc))

    def export_csv(self) -> None:
        from tkinter import filedialog, messagebox

        if self.current_result is None:
            messagebox.showwarning("Ni rezultatov", "Najprej izvedi izračun.")
            return
        rtp = safe_name(self.rtp_var.get())
        selected = filedialog.asksaveasfilename(
            title="Izvozi kompenzirane rezultate",
            initialfile=f"{rtp}_kompenzacija_jalove_moci.csv",
            defaultextension=".csv",
            filetypes=[("CSV datoteka", "*.csv")],
        )
        if selected:
            try:
                self.current_result.to_csv(selected, index=False, encoding="utf-8-sig")
                self.status_var.set(f"Rezultati izvoženi: {selected}")
            except Exception as exc:
                messagebox.showerror("Napaka pri izvozu", str(exc))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interaktivni izračun kompenzacije jalove moči po RTP."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_directory(),
        help="Mapa component_files s transformatorskimi Parquet datotekami.",
    )
    parser.add_argument("--smoke-test", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--rtp", help=argparse.SUPPRESS)
    parser.add_argument("--capacitive", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--inductive", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def run_smoke_test(args: argparse.Namespace) -> None:
    rtp_files, stats = discover_rtp_files(args.data_dir.resolve())
    if not rtp_files:
        raise RuntimeError("Ni veljavnih RTP.")
    rtp = args.rtp or next(iter(rtp_files))
    if rtp not in rtp_files:
        raise RuntimeError(f"RTP '{rtp}' ne obstaja. Možnosti: {', '.join(rtp_files)}")
    source = load_rtp_power(rtp_files[rtp])
    original = calculate_compensation(
        source,
        capacitive_mvar=0,
        inductive_mvar=0,
        cosphi_limit=COSPHI_LIMIT_DEFAULT,
        use_abs_p=False,
    )
    result = calculate_compensation(
        source,
        capacitive_mvar=args.capacitive,
        inductive_mvar=args.inductive,
        cosphi_limit=COSPHI_LIMIT_DEFAULT,
        use_abs_p=False,
    )
    statistics = scenario_statistics(result)
    print(f"RTP: {rtp}")
    print(f"Najdenih RTP: {stats['rtp_count']}")
    print(f"Transformatorjev v RTP: {len(rtp_files[rtp])}")
    print(f"Merilnih točk: {statistics['count']}")
    print(
        "Kapacitivno/v mejah/induktivno: "
        f"{statistics['pct_cap']:.2f}% / {statistics['pct_ok']:.2f}% / "
        f"{statistics['pct_ind']:.2f}%"
    )
    if args.output:
        figure = build_figure(
            rtp,
            original,
            result,
            capacitive_mvar=args.capacitive,
            inductive_mvar=args.inductive,
            cosphi_limit=COSPHI_LIMIT_DEFAULT,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output, bbox_inches="tight")
        print(f"Graf: {args.output.resolve()}")


def main() -> None:
    args = parse_arguments()
    if args.smoke_test:
        run_smoke_test(args)
        return

    import tkinter as tk

    root = tk.Tk()
    CompensationApp(root, args.data_dir.resolve())
    root.mainloop()


if __name__ == "__main__":
    main()
