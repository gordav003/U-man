from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, time
from pathlib import Path
from typing import Any


APP_TITLE = "Izbirnik Parquet grafov"
TIME_COLUMN = "time"
QUALITY_FILTERED_TAP_COLUMNS = {"TAP"}
PREFERRED_MEASUREMENTS = ("U", "P", "Q", "TAP")
MEASUREMENT_LABELS = {
    "U": "Napetost U [kV]",
    "P": "Delovna moč P [MW]",
    "Q": "Jalova moč Q [MVAr]",
    "TAP": "Položaj regulatorja TAP",
}
MEASUREMENT_UNITS = {
    "U": "kV",
    "P": "MW",
    "Q": "MVAr",
    "TAP": "",
}


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_data_directories() -> list[Path]:
    prepared = (
        project_root()
        / "Uman meritve"
        / "2026_06_17  SCADA meritve 4600"
        / "urejeno"
    )
    return [
        prepared / "Uman_parquet" / "component_files",
        prepared / "Uman_TAP_parquet" / "tap_component_files",
    ]


def import_polars():
    try:
        import polars as pl
    except ImportError as exc:
        raise RuntimeError(
            "Manjka knjižnica 'polars'. Namesti odvisnosti z ukazom:\n"
            "python -m pip install -r requirements.txt"
        ) from exc
    return pl


@dataclass(frozen=True)
class PlotRequest:
    file: str
    measurement: str
    start: str
    end: str
    title: str = ""

    @property
    def path(self) -> Path:
        return Path(self.file)


def component_type(path: Path) -> str:
    prefix = path.stem.split("_", 1)[0].upper()
    labels = {
        "TR": "Transformatorji",
        "LINE": "Daljnovodi",
        "GEN": "Generatorji",
        "G": "Generatorji",
        "TAP": "Regulatorji TAP",
        "METER": "Merilna mesta",
        "BLOK": "Bloki",
        "PB": "Črpalne elektrarne",
        "BHEE": "Baterije / HEE",
    }
    return labels.get(prefix, f"Drugo ({prefix})")


def display_component_name(path: Path) -> str:
    return path.stem.replace("_", " ")


def component_plot_label(path: Path, *, include_asset: bool = False) -> str:
    """Vrne kratko oznako lokacije za navpično oznako ob grafu."""
    parts = path.stem.split("_")
    if len(parts) < 2:
        return path.stem

    component_kind = parts[0].upper()
    body = parts[1:]
    voltage_index = next(
        (index for index, part in enumerate(body) if part.isdigit()),
        None,
    )

    if voltage_index is None:
        return " ".join(body)

    location_from = " ".join(body[:voltage_index])
    after_voltage = body[voltage_index + 1 :]

    if component_kind == "LINE" and after_voltage:
        destination_parts = after_voltage[:-1] if after_voltage[-1].isdigit() else after_voltage
        destination = " ".join(destination_parts)
        return (
            f"{location_from}–{destination}"
            if destination
            else location_from
        )

    if component_kind == "TR":
        asset = " ".join(after_voltage)
        if include_asset and asset:
            return f"{location_from}\n{asset}"
        return location_from

    return location_from or " ".join(body)


def discover_parquet_files(directories: list[Path]) -> list[Path]:
    found: dict[str, Path] = {}
    for directory in directories:
        if not directory.is_dir():
            continue
        for path in directory.rglob("*.parquet"):
            resolved = path.resolve()
            found[str(resolved).casefold()] = resolved
    return sorted(
        found.values(),
        key=lambda item: (component_type(item), item.stem.casefold()),
    )


def parse_date_text(value: str, *, end_of_day: bool) -> datetime:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("Datum ne sme biti prazen.")

    formats = (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%d.%m.%Y %H:%M:%S",
        "%d.%m.%Y %H:%M",
        "%d.%m.%Y",
    )
    parsed: datetime | None = None
    used_date_only = False
    for date_format in formats:
        try:
            parsed = datetime.strptime(cleaned, date_format)
            used_date_only = date_format in {"%Y-%m-%d", "%d.%m.%Y"}
            break
        except ValueError:
            continue

    if parsed is None:
        raise ValueError(
            f"Datuma '{value}' ne prepoznam. Uporabi npr. 2025-01-01 "
            "ali 2025-01-01 06:00."
        )
    if end_of_day and used_date_only:
        return datetime.combine(parsed.date(), time.max)
    return parsed


def available_measurements(path: Path) -> list[str]:
    pl = import_polars()
    lazy = pl.scan_parquet(path)
    schema = lazy.collect_schema()
    columns = set(schema.names())
    numeric_types = {
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
    }
    candidates = [
        name for name in PREFERRED_MEASUREMENTS if name in columns
    ] or [
        name
        for name, dtype in schema.items()
        if name != TIME_COLUMN and dtype in numeric_types
    ]
    if not candidates:
        return []

    availability = (
        lazy.select(
            *[
                pl.col(name).is_not_null().any().alias(name)
                for name in candidates
            ]
        )
        .collect()
        .row(0, named=True)
    )
    return [name for name in candidates if availability[name]]


def data_time_range(path: Path) -> tuple[datetime, datetime]:
    pl = import_polars()
    result = (
        pl.scan_parquet(path)
        .select(
            pl.col(TIME_COLUMN).min().alias("start"),
            pl.col(TIME_COLUMN).max().alias("end"),
        )
        .collect()
        .row(0, named=True)
    )
    if result["start"] is None or result["end"] is None:
        raise ValueError(f"Datoteka {path.name} nima veljavnih časovnih podatkov.")
    return result["start"], result["end"]


def measurement_time_range(
    path: Path,
    measurement: str,
) -> tuple[datetime, datetime]:
    pl = import_polars()
    result = (
        pl.scan_parquet(path)
        .filter(pl.col(measurement).is_not_null())
        .select(
            pl.col(TIME_COLUMN).min().alias("start"),
            pl.col(TIME_COLUMN).max().alias("end"),
        )
        .collect()
        .row(0, named=True)
    )
    if result["start"] is None or result["end"] is None:
        raise ValueError(
            f"Datoteka {path.name} nima veljavnih vrednosti za {measurement}."
        )
    return result["start"], result["end"]


def load_plot_data(request: PlotRequest) -> tuple[list[datetime], list[float | None]]:
    pl = import_polars()
    start = parse_date_text(request.start, end_of_day=False)
    end = parse_date_text(request.end, end_of_day=True)
    if start > end:
        raise ValueError("Začetek obdobja mora biti pred koncem obdobja.")

    lazy = pl.scan_parquet(request.path)
    schema_names = set(lazy.collect_schema().names())
    missing = {TIME_COLUMN, request.measurement} - schema_names
    if missing:
        raise ValueError(
            f"{request.path.name}: manjkajo stolpci {', '.join(sorted(missing))}."
        )

    if (
        request.measurement in QUALITY_FILTERED_TAP_COLUMNS
        and {"qst_no", "parse_status"}.issubset(schema_names)
    ):
        lazy = lazy.filter(
            (pl.col("qst_no") == 1)
            & (pl.col("parse_status").str.to_lowercase() == "ok")
        )

    data = (
        lazy.filter(pl.col(TIME_COLUMN).is_between(start, end, closed="both"))
        .select(TIME_COLUMN, request.measurement)
        .drop_nulls(subset=[TIME_COLUMN])
        .group_by(TIME_COLUMN)
        .agg(pl.col(request.measurement).mean())
        .sort(TIME_COLUMN)
        .collect()
    )
    return (
        data.get_column(TIME_COLUMN).to_list(),
        data.get_column(request.measurement).to_list(),
    )


def save_configuration(path: Path, requests: list[PlotRequest]) -> None:
    payload = {
        "version": 1,
        "plots": [asdict(request) for request in requests],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_configuration(path: Path) -> list[PlotRequest]:
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("plots") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("Konfiguracija nima seznama 'plots'.")

    requests = []
    for row in rows:
        request = PlotRequest(**row)
        if not request.path.is_file():
            raise FileNotFoundError(
                f"Datoteka iz konfiguracije ne obstaja: {request.path}"
            )
        requests.append(request)
    return requests


def build_plot_figure(requests: list[PlotRequest]):
    from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator

    try:
        periods = {
            (
                parse_date_text(request.start, end_of_day=False),
                parse_date_text(request.end, end_of_day=True),
            )
            for request in requests
        }
        shared_time_axis = len(periods) == 1
    except ValueError:
        shared_time_axis = False

    figure = Figure(
        figsize=(10, max(3.2, 1.8 * len(requests))),
        constrained_layout=True,
        facecolor="white",
    )
    axes = figure.subplots(
        nrows=len(requests),
        ncols=1,
        squeeze=False,
        sharex=shared_time_axis,
    ).ravel()
    errors: list[str] = []
    location_counts = Counter(
        component_plot_label(request.path)
        for request in requests
    )

    for axis, request in zip(axes, requests):
        basic_location = component_plot_label(request.path)
        location = component_plot_label(
            request.path,
            include_asset=location_counts[basic_location] > 1,
        )
        try:
            timestamps, values = load_plot_data(request)
            valid_count = sum(value is not None for value in values)
            if not timestamps or valid_count == 0:
                axis.text(
                    0.5,
                    0.5,
                    "V izbranem obdobju ni podatkov.",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                )
            else:
                axis.plot(
                    timestamps,
                    values,
                    linewidth=1.25,
                    color="#145da0",
                )

            unit = MEASUREMENT_UNITS.get(request.measurement, "")
            axis.set_ylabel(
                location,
                rotation=90,
                fontsize=9,
                fontweight="bold",
                color="#263238",
                labelpad=12,
            )
            measurement_label = (
                f"{request.measurement} [{unit}]"
                if unit
                else request.measurement
            )
            axis.text(
                0.006,
                0.92,
                measurement_label,
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color="#455a64",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.78,
                    "pad": 1.5,
                },
            )
            axis.set_facecolor("white")
            axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
            axis.grid(axis="y", color="#cfd8dc", linewidth=0.7, alpha=0.7)
            axis.grid(axis="x", color="#eceff1", linewidth=0.6, alpha=0.8)
            axis.tick_params(
                axis="x",
                labelrotation=0,
                labelsize=8,
                colors="#455a64",
                length=3,
            )
            axis.tick_params(
                axis="y",
                labelsize=8,
                colors="#455a64",
                length=3,
            )
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.spines["left"].set_color("#90a4ae")
            axis.spines["bottom"].set_color("#90a4ae")
            locator = AutoDateLocator(minticks=3, maxticks=7)
            axis.xaxis.set_major_locator(locator)
            axis.xaxis.set_major_formatter(ConciseDateFormatter(locator))
        except Exception as exc:
            errors.append(f"{request.path.name}: {exc}")
            axis.text(
                0.5,
                0.5,
                f"Napaka:\n{exc}",
                transform=axis.transAxes,
                ha="center",
                va="center",
                color="darkred",
                wrap=True,
            )
            axis.set_ylabel(
                location,
                rotation=90,
                fontsize=9,
                fontweight="bold",
                labelpad=12,
            )

    if shared_time_axis and len(axes) > 1:
        for axis in axes[:-1]:
            axis.tick_params(axis="x", labelbottom=False)

    return figure, errors


def launch_app(initial_directories: list[Path]) -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    try:
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg,
            NavigationToolbar2Tk,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Manjka knjižnica 'matplotlib'. Namesti odvisnosti z ukazom:\n"
            "python -m pip install -r requirements.txt"
        ) from exc

    class PlotterApp:
        def __init__(self, root: tk.Tk) -> None:
            self.root = root
            self.root.title(APP_TITLE)
            self.root.geometry("1380x860")
            self.root.minsize(1050, 680)

            self.directories = [
                path.resolve() for path in initial_directories if path.is_dir()
            ]
            self.files: list[Path] = []
            self.filtered_files: list[Path] = []
            self.requests: list[PlotRequest] = []
            self.measurement_vars: dict[str, tk.BooleanVar] = {}
            self.figure: Figure | None = None
            self.canvas: FigureCanvasTkAgg | None = None
            self.toolbar: NavigationToolbar2Tk | None = None

            self.search_var = tk.StringVar()
            self.type_var = tk.StringVar(value="Vse vrste")
            self.period_info_var = tk.StringVar(value="Izberi komponento.")
            self.start_var = tk.StringVar()
            self.end_var = tk.StringVar()
            self.status_var = tk.StringVar(value="Pripravljeno.")

            self._build_ui()
            self.refresh_files()

        def _build_ui(self) -> None:
            style = ttk.Style()
            style.configure("Title.TLabel", font=("Segoe UI", 17, "bold"))
            style.configure("Section.TLabel", font=("Segoe UI", 10, "bold"))

            main = ttk.Frame(self.root, padding=12)
            main.pack(fill=tk.BOTH, expand=True)
            main.columnconfigure(0, weight=1, minsize=360)
            main.columnconfigure(1, weight=2)
            main.rowconfigure(1, weight=1)

            heading = ttk.Frame(main)
            heading.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
            ttk.Label(
                heading,
                text="Poljubni grafi iz Parquet meritev",
                style="Title.TLabel",
            ).pack(side=tk.LEFT)
            ttk.Label(
                heading,
                text="Vsak dodani graf ima svojo komponento, meritev in obdobje.",
            ).pack(side=tk.LEFT, padx=18, pady=(7, 0))

            controls = ttk.Frame(main)
            controls.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
            controls.columnconfigure(0, weight=1)
            controls.rowconfigure(3, weight=1)

            source = ttk.LabelFrame(controls, text="1. Podatkovni vir", padding=8)
            source.grid(row=0, column=0, sticky="ew")
            source.columnconfigure(0, weight=1)
            self.folder_label = ttk.Label(
                source,
                text="",
                wraplength=330,
                foreground="#444444",
            )
            self.folder_label.grid(row=0, column=0, sticky="w")
            ttk.Button(
                source,
                text="Dodaj mapo ...",
                command=self.add_directory,
            ).grid(row=1, column=0, sticky="w", pady=(6, 0))
            ttk.Button(
                source,
                text="Ponovno preglej",
                command=self.refresh_files,
            ).grid(row=1, column=0, sticky="e", pady=(6, 0))

            filters = ttk.LabelFrame(controls, text="2. Komponenta", padding=8)
            filters.grid(row=1, column=0, sticky="ew", pady=(8, 0))
            filters.columnconfigure(0, weight=1)
            ttk.Entry(filters, textvariable=self.search_var).grid(
                row=0, column=0, sticky="ew"
            )
            self.type_combo = ttk.Combobox(
                filters,
                textvariable=self.type_var,
                state="readonly",
            )
            self.type_combo.grid(row=1, column=0, sticky="ew", pady=(6, 0))
            self.search_var.trace_add("write", lambda *_: self.apply_filters())
            self.type_combo.bind("<<ComboboxSelected>>", lambda *_: self.apply_filters())

            file_frame = ttk.Frame(controls)
            file_frame.grid(row=3, column=0, sticky="nsew", pady=(8, 0))
            file_frame.columnconfigure(0, weight=1)
            file_frame.rowconfigure(0, weight=1)
            self.file_list = tk.Listbox(
                file_frame,
                exportselection=False,
                font=("Consolas", 9),
            )
            self.file_list.grid(row=0, column=0, sticky="nsew")
            scrollbar = ttk.Scrollbar(
                file_frame,
                orient=tk.VERTICAL,
                command=self.file_list.yview,
            )
            scrollbar.grid(row=0, column=1, sticky="ns")
            self.file_list.configure(yscrollcommand=scrollbar.set)
            self.file_list.bind("<<ListboxSelect>>", self.on_file_selected)

            options = ttk.LabelFrame(controls, text="3. Meritev in obdobje", padding=8)
            options.grid(row=4, column=0, sticky="ew", pady=(8, 0))
            options.columnconfigure(1, weight=1)
            self.measurements_frame = ttk.Frame(options)
            self.measurements_frame.grid(
                row=0, column=0, columnspan=2, sticky="ew"
            )
            ttk.Label(
                options,
                textvariable=self.period_info_var,
                foreground="#555555",
            ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 7))
            ttk.Label(options, text="Od:").grid(row=2, column=0, sticky="w")
            ttk.Entry(options, textvariable=self.start_var).grid(
                row=2, column=1, sticky="ew"
            )
            ttk.Label(options, text="Do:").grid(row=3, column=0, sticky="w")
            ttk.Entry(options, textvariable=self.end_var).grid(
                row=3, column=1, sticky="ew", pady=(4, 0)
            )
            ttk.Button(
                options,
                text="Dodaj izbrane meritve v zbirko",
                command=self.add_requests,
            ).grid(row=4, column=0, columnspan=2, sticky="ew", pady=(8, 0))

            right = ttk.Frame(main)
            right.grid(row=1, column=1, sticky="nsew")
            right.columnconfigure(0, weight=1)
            right.rowconfigure(1, weight=1)

            queue_frame = ttk.LabelFrame(right, text="Zbirka grafov", padding=8)
            queue_frame.grid(row=0, column=0, sticky="ew")
            queue_frame.columnconfigure(0, weight=1)
            columns = ("measurement", "component", "start", "end")
            self.queue = ttk.Treeview(
                queue_frame,
                columns=columns,
                show="headings",
                height=6,
                selectmode="extended",
            )
            headings = {
                "measurement": "Meritev",
                "component": "Komponenta",
                "start": "Od",
                "end": "Do",
            }
            widths = {
                "measurement": 80,
                "component": 300,
                "start": 125,
                "end": 125,
            }
            for column in columns:
                self.queue.heading(column, text=headings[column])
                self.queue.column(column, width=widths[column], anchor="w")
            self.queue.grid(row=0, column=0, columnspan=6, sticky="ew")

            buttons = (
                ("Odstrani", self.remove_requests),
                ("Počisti", self.clear_requests),
                ("Shrani izbor ...", self.save_requests),
                ("Odpri izbor ...", self.open_requests),
                ("Izriši zbirko", self.draw_requests),
                ("Shrani sliko ...", self.save_figure),
            )
            for index, (label, command) in enumerate(buttons):
                ttk.Button(queue_frame, text=label, command=command).grid(
                    row=1, column=index, padx=(0 if index == 0 else 5, 0), pady=(7, 0)
                )

            self.plot_frame = ttk.LabelFrame(right, text="Predogled", padding=5)
            self.plot_frame.grid(row=1, column=0, sticky="nsew", pady=(8, 0))

            status = ttk.Label(
                main,
                textvariable=self.status_var,
                relief=tk.SUNKEN,
                anchor="w",
                padding=4,
            )
            status.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        def refresh_files(self) -> None:
            self.files = discover_parquet_files(self.directories)
            types = sorted({component_type(path) for path in self.files})
            self.type_combo["values"] = ["Vse vrste", *types]
            if self.type_var.get() not in self.type_combo["values"]:
                self.type_var.set("Vse vrste")
            self.folder_label.configure(
                text="\n".join(str(path) for path in self.directories)
            )
            self.apply_filters()
            self.status_var.set(f"Najdenih Parquet datotek: {len(self.files)}.")

        def add_directory(self) -> None:
            selected = filedialog.askdirectory(title="Izberi mapo s Parquet datotekami")
            if not selected:
                return
            path = Path(selected).resolve()
            if path not in self.directories:
                self.directories.append(path)
            self.refresh_files()

        def apply_filters(self) -> None:
            query = self.search_var.get().strip().casefold()
            selected_type = self.type_var.get()
            self.filtered_files = [
                path
                for path in self.files
                if (not query or query in path.stem.casefold())
                and (
                    selected_type == "Vse vrste"
                    or component_type(path) == selected_type
                )
            ]
            self.file_list.delete(0, tk.END)
            for path in self.filtered_files:
                self.file_list.insert(
                    tk.END,
                    f"[{component_type(path)}] {display_component_name(path)}",
                )

        def selected_file(self) -> Path | None:
            selection = self.file_list.curselection()
            if not selection:
                return None
            return self.filtered_files[selection[0]]

        def on_file_selected(self, _event: object = None) -> None:
            path = self.selected_file()
            if path is None:
                return
            try:
                measurements = available_measurements(path)
                start, end = data_time_range(path)
            except Exception as exc:
                messagebox.showerror("Napaka pri branju", str(exc))
                return

            for widget in self.measurements_frame.winfo_children():
                widget.destroy()
            self.measurement_vars = {}
            for index, measurement in enumerate(measurements):
                variable = tk.BooleanVar(value=(index == 0))
                self.measurement_vars[measurement] = variable
                ttk.Checkbutton(
                    self.measurements_frame,
                    text=MEASUREMENT_LABELS.get(measurement, measurement),
                    variable=variable,
                ).pack(side=tk.LEFT, padx=(0, 10))

            self.start_var.set(start.strftime("%Y-%m-%d"))
            self.end_var.set(end.strftime("%Y-%m-%d"))
            self.period_info_var.set(
                f"Razpoložljivo: {start:%d.%m.%Y %H:%M} – "
                f"{end:%d.%m.%Y %H:%M}"
            )
            self.status_var.set(f"Izbrano: {path.name}")

        def add_requests(self) -> None:
            path = self.selected_file()
            if path is None:
                messagebox.showwarning("Ni komponente", "Najprej izberi komponento.")
                return
            measurements = [
                name for name, variable in self.measurement_vars.items() if variable.get()
            ]
            if not measurements:
                messagebox.showwarning("Ni meritve", "Obkljukaj vsaj eno meritev.")
                return
            try:
                start = parse_date_text(self.start_var.get(), end_of_day=False)
                end = parse_date_text(self.end_var.get(), end_of_day=True)
                if start > end:
                    raise ValueError("Začetek obdobja mora biti pred koncem obdobja.")
            except ValueError as exc:
                messagebox.showerror("Napačno obdobje", str(exc))
                return

            for measurement in measurements:
                self.requests.append(
                    PlotRequest(
                        file=str(path),
                        measurement=measurement,
                        start=self.start_var.get().strip(),
                        end=self.end_var.get().strip(),
                    )
                )
            self.refresh_queue()
            self.status_var.set(
                f"Dodano: {len(measurements)}. V zbirki: {len(self.requests)} grafov."
            )

        def refresh_queue(self) -> None:
            self.queue.delete(*self.queue.get_children())
            for index, request in enumerate(self.requests):
                self.queue.insert(
                    "",
                    tk.END,
                    iid=str(index),
                    values=(
                        request.measurement,
                        display_component_name(request.path),
                        request.start,
                        request.end,
                    ),
                )

        def remove_requests(self) -> None:
            indices = sorted(
                (int(item) for item in self.queue.selection()),
                reverse=True,
            )
            for index in indices:
                del self.requests[index]
            self.refresh_queue()

        def clear_requests(self) -> None:
            self.requests.clear()
            self.refresh_queue()
            self.status_var.set("Zbirka je prazna.")

        def save_requests(self) -> None:
            if not self.requests:
                messagebox.showwarning("Prazna zbirka", "Ni izbora za shranjevanje.")
                return
            selected = filedialog.asksaveasfilename(
                title="Shrani izbor grafov",
                defaultextension=".json",
                filetypes=[("JSON konfiguracija", "*.json")],
            )
            if not selected:
                return
            try:
                save_configuration(Path(selected), self.requests)
                self.status_var.set(f"Izbor shranjen: {selected}")
            except Exception as exc:
                messagebox.showerror("Napaka pri shranjevanju", str(exc))

        def open_requests(self) -> None:
            selected = filedialog.askopenfilename(
                title="Odpri izbor grafov",
                filetypes=[("JSON konfiguracija", "*.json")],
            )
            if not selected:
                return
            try:
                self.requests = load_configuration(Path(selected))
                self.refresh_queue()
                self.status_var.set(
                    f"Naložena konfiguracija z {len(self.requests)} grafi."
                )
            except Exception as exc:
                messagebox.showerror("Napaka pri odpiranju", str(exc))

        def _clear_plot_frame(self) -> None:
            if self.toolbar is not None:
                self.toolbar.destroy()
                self.toolbar = None
            if self.canvas is not None:
                self.canvas.get_tk_widget().destroy()
                self.canvas = None

        def draw_requests(self) -> None:
            if not self.requests:
                messagebox.showwarning(
                    "Prazna zbirka",
                    "Dodaj vsaj en graf v zbirko.",
                )
                return

            self.status_var.set("Berem podatke in pripravljam grafe ...")
            self.root.update_idletasks()
            figure, errors = build_plot_figure(self.requests)

            self._clear_plot_frame()
            self.figure = figure
            self.canvas = FigureCanvasTkAgg(figure, master=self.plot_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self.toolbar = NavigationToolbar2Tk(
                self.canvas,
                self.plot_frame,
                pack_toolbar=False,
            )
            self.toolbar.update()
            self.toolbar.pack(fill=tk.X)
            self.status_var.set(
                f"Izrisano: {len(self.requests) - len(errors)} od "
                f"{len(self.requests)} grafov."
            )
            if errors:
                messagebox.showwarning(
                    "Nekaterih grafov ni bilo mogoče izrisati",
                    "\n\n".join(errors[:8]),
                )

        def save_figure(self) -> None:
            if self.figure is None:
                messagebox.showwarning(
                    "Ni slike",
                    "Najprej klikni 'Izriši zbirko'.",
                )
                return
            selected = filedialog.asksaveasfilename(
                title="Shrani zbirko grafov",
                defaultextension=".png",
                filetypes=[
                    ("PNG slika", "*.png"),
                    ("SVG vektorska slika", "*.svg"),
                    ("PDF dokument", "*.pdf"),
                ],
            )
            if not selected:
                return
            try:
                self.figure.savefig(selected, dpi=180, bbox_inches="tight")
                self.status_var.set(f"Slika shranjena: {selected}")
            except Exception as exc:
                messagebox.showerror("Napaka pri shranjevanju slike", str(exc))

    root = tk.Tk()
    PlotterApp(root)
    root.mainloop()


def smoke_test(directories: list[Path]) -> None:
    files = discover_parquet_files(directories)
    if not files:
        raise RuntimeError("V nastavljenih mapah ni Parquet datotek.")

    chosen = next(
        (
            path
            for path in files
            if path.name.startswith("TR_")
            and {"U", "P", "Q"}.intersection(available_measurements(path))
        ),
        files[0],
    )
    measurements = available_measurements(chosen)
    if not measurements:
        raise RuntimeError(f"{chosen.name} nima merilnih stolpcev.")
    start, end = measurement_time_range(chosen, measurements[0])
    request = PlotRequest(
        file=str(chosen),
        measurement=measurements[0],
        start=start.strftime("%Y-%m-%d"),
        end=min(end, start.replace(hour=23, minute=59)).strftime(
            "%Y-%m-%d %H:%M"
        ),
    )
    timestamps, values = load_plot_data(request)
    print(f"Parquet datotek: {len(files)}")
    print(f"Preizkusna datoteka: {chosen}")
    print(f"Meritve: {', '.join(measurements)}")
    print(f"Časovni razpon: {start} – {end}")
    print(f"Prebranih točk v preizkusu: {len(timestamps)}")
    if not timestamps or not any(value is not None for value in values):
        raise RuntimeError("Preizkusno branje ni vrnilo veljavnih točk.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Namizni izbirnik za poljubno zbiranje in izris meritev iz "
            "Parquet datotek."
        )
    )
    parser.add_argument(
        "--data-dir",
        action="append",
        type=Path,
        help=(
            "Mapa s Parquet datotekami; možnost lahko podaš večkrat. "
            "Brez nje se uporabijo projektne mape component_files in TAP."
        ),
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Preveri odkrivanje datotek in eno dejansko branje brez odprtja okna.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    directories = args.data_dir or default_data_directories()
    try:
        if args.smoke_test:
            smoke_test(directories)
        else:
            launch_app(directories)
    except Exception as exc:
        print(f"NAPAKA: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
