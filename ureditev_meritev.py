from pathlib import Path
import re
import polars as pl
import csv

# NASTAVITVE

INPUT_DIR = Path(r"C:\LEON\Projekti\2026\CRESYM-Uman\Uman meritve\2026_06_17  SCADA meritve 4600")
OUT_DIR = Path(r"C:\LEON\Projekti\2026\CRESYM-Uman\Uman meritve\2026_06_17  SCADA meritve 4600\urejeno\Uman_parquet")

#MODE = "metadata"
#MODE = "problems"
#MODE = "normalize"
#MODE = "wide"
#MODE = "component_catalog"
MODE = "component_files"

OUT_DIR.mkdir(parents=True, exist_ok=True)

METADATA_FILE = OUT_DIR / "metadata_review.csv"
PARQUET_DIR = OUT_DIR / "parquet_normalized"
PARQUET_DIR.mkdir(parents=True, exist_ok=True)
COMPONENT_EXPORT_DIR = OUT_DIR / "component_files"
COMPONENT_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_ONLY_QST_1 = True

VOLTAGES = ["400", "220", "110", "35", "21", "20", "18", "13", "11", "10", "6", "5"]
VOLTAGES = sorted(VOLTAGES, key=len, reverse=True)
VOLTAGE_RE = "|".join(VOLTAGES)

MEASUREMENTS = {"P", "Q", "U", "T"}

ONLY_QST_1 = False

# CSV BRANJE

BASE_COLUMNS = [
    "exid",
    "systime",
    "value",
    "qst_no",
    "timestamp",
    "calc_counter",
]


def read_first_nonempty_lines(path: Path, n: int = 50):
    lines = []

    with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
        for line in f:
            line = line.strip("\n\r")

            if line.strip():
                lines.append(line)

            if len(lines) >= n:
                break

    return lines


def count_fields(line: str, sep: str) -> int:
    try:
        return len(next(csv.reader([line], delimiter=sep)))
    except Exception:
        return 0


def sniff_separator(path: Path) -> str:
    """
    Ne uporabljamo več logike 'največ znakov zmaga',
    ker decimalna vejica lahko pokvari zaznavo.

    Izberemo separator, ki daje stabilno število stolpcev.
    """
    lines = read_first_nonempty_lines(path, n=50)

    if not lines:
        raise ValueError(f"Prazna datoteka: {path}")

    candidates = ["\t", ";", "|", ","]

    best_sep = None
    best_score = None

    for sep in candidates:
        counts = [count_fields(line, sep) for line in lines]
        valid_counts = [c for c in counts if c >= 5]

        if not valid_counts:
            continue

        # več vrstic z vsaj 5 stolpci je bolje
        # manj različnih dolžin je bolje
        # večja tipična širina je bolje
        score = (
            len(valid_counts),
            -len(set(valid_counts)),
            max(valid_counts),
        )

        if best_score is None or score > best_score:
            best_score = score
            best_sep = sep

    if best_sep is None:
        raise ValueError(f"Ne morem zaznati separatorja za: {path}")

    return best_sep


def has_header(path: Path, sep: str) -> bool:
    lines = read_first_nonempty_lines(path, n=1)

    if not lines:
        return False

    first_line = lines[0]
    fields = next(csv.reader([first_line], delimiter=sep))

    fields_clean = [
        f.replace("\ufeff", "").strip().lower().replace(" ", "")
        for f in fields
    ]

    return (
        "exid" in fields_clean
        or "systime(utc+1)" in fields_clean
        or "value" in fields_clean
    )


def detect_num_columns(path: Path, sep: str, skip_rows: int = 0, n: int = 100) -> int:
    lines = read_first_nonempty_lines(path, n=n + skip_rows)

    if skip_rows > 0:
        lines = lines[skip_rows:]

    if not lines:
        return len(BASE_COLUMNS)

    counts = [count_fields(line, sep) for line in lines]
    n_cols = max(counts)

    return max(n_cols, len(BASE_COLUMNS))


def make_column_names(n_cols: int):
    """
    Če ima datoteka več stolpcev kot osnovnih 6,
    jih ohranimo kot extra_1, extra_2 ...
    """
    names = BASE_COLUMNS.copy()

    if n_cols > len(BASE_COLUMNS):
        for i in range(n_cols - len(BASE_COLUMNS)):
            names.append(f"extra_{i + 1}")

    return names


def scan_csv_clean(path: Path) -> pl.LazyFrame:
    sep = sniff_separator(path)
    header_exists = has_header(path, sep)

    skip_rows = 1 if header_exists else 0

    n_cols = detect_num_columns(
        path=path,
        sep=sep,
        skip_rows=skip_rows,
        n=100,
    )

    column_names = make_column_names(n_cols)

    print(
        f"{path.name} | sep={repr(sep)} | "
        f"header={header_exists} | stolpci={n_cols}"
    )

    lf = pl.scan_csv(
        path,
        separator=sep,
        has_header=False,
        skip_rows=skip_rows,
        new_columns=column_names,
        infer_schema_length=0,
        encoding="utf8-lossy",
        ignore_errors=True,
        null_values=["", "NULL", "NaN"],
        truncate_ragged_lines=True,
    )

    return lf

def split_measurement(x: str):
    """
    Normalni primeri:
    MARIBOR400TR411P -> body=MARIBOR400TR411, meritev=P, suffix=None

    Posebni primeri:
    FORMIN110G1UA -> body=FORMIN110G1, meritev=U, suffix=A
    AJDOVSCINA110MERITVE2UB -> body=AJDOVSCINA110MERITVE2, meritev=U, suffix=B
    """

    if len(x) >= 2:
        second_last = x[-2]
        last = x[-1]

        if second_last in MEASUREMENTS and last.isalpha() and last not in MEASUREMENTS:
            return x[:-2], second_last, last

    if x and x[-1] in MEASUREMENTS:
        return x[:-1], x[-1], None

    return x, None, None

# PARSER EXID

def split_line_tail(tail: str):
    """
    CIRKOVCE2        -> CIRKOVCE, 2
    O_KAINACH474     -> O_KAINACH, 474
    H_TUMBRI1        -> H_TUMBRI, 1
    I_REDIPUGLIA     -> I_REDIPUGLIA, None
    CERKNO           -> CERKNO, None
    """

    tail = tail.strip()

    m = re.match(r"^(?P<name>.*?)(?P<number>\d+)$", tail)

    if m:
        return m.group("name"), m.group("number")

    return tail, None


def parse_exid(exid: str) -> dict:
    original = str(exid).strip()
    x = original.upper()

    out = {
        "exid_key": x,
        "exid_original": original,
        "lokacija_od": None,
        "napetost_kv": None,
        "tip_objekta": "unknown",
        "objekt": None,
        "lokacija_do": None,
        "oznaka": None,
        "meritev": None,
        "meritev_suffix": None,
        "component_id": None,
        "parse_status": "unparsed",
    }

    if len(x) < 2:
        return out

    # 1) Loči meritev: P/Q/U/T ali U + suffix A/B
    body, meritev, meritev_suffix = split_measurement(x)

    if meritev is None:
        return out

    out["meritev"] = meritev
    out["meritev_suffix"] = meritev_suffix

    # 2) Poseben primer: tap meritve brez napetostnega nivoja
    # Primeri:
    # AJDOVSCINATR1T -> AJDOVSCINA, TR1, T
    # BERICEVOT211T -> BERICEVO, T211, T
    # CIRKOVCEL401T -> CIRKOVCE, L401, T
    # SEZANATRAT -> SEZANA, TRA, T
    if meritev == "T":
        m_tap = re.match(
            r"^(?P<lokacija_od>.+)(?P<objekt>TR[A-Z0-9]*|T\d+[A-Z]*|L\d+[A-Z]*)$",
            body
        )

        if m_tap:
            lokacija_od = m_tap.group("lokacija_od")
            objekt = m_tap.group("objekt")

            out.update({
                "lokacija_od": lokacija_od,
                "tip_objekta": "tap",
                "objekt": objekt,
                "component_id": f"TAP|{lokacija_od}|{objekt}",
                "parse_status": "ok",
            })

            return out

    # 3) Klasični vzorec: lokacija + napetost + tail
    m = re.match(
        rf"^(?P<lokacija_od>.*?)(?P<napetost>{VOLTAGE_RE})(?P<tail>.+)$",
        body
    )

    if not m:
        return out

    lokacija_od = m.group("lokacija_od")
    napetost = int(m.group("napetost"))
    tail = m.group("tail")

    out["lokacija_od"] = lokacija_od
    out["napetost_kv"] = napetost
    out["objekt"] = tail

    # 4) Transformatorji
    if tail.startswith("TR"):
        out["tip_objekta"] = "transformer_tap" if meritev == "T" else "transformer"
        out["component_id"] = f"TR|{lokacija_od}|{napetost}|{tail}"
        out["parse_status"] = "ok"
        return out

    # 5) Generatorji GEN
    if tail.startswith("GEN"):
        out["tip_objekta"] = "generator"
        out["component_id"] = f"GEN|{lokacija_od}|{napetost}|{tail}"
        out["parse_status"] = "ok"
        return out

    # 6) G1, G2, G3 ...
    # To pustimo kot review, ker lahko pomeni generator ali polje.
    if re.match(r"^G\d+$", tail):
        out["tip_objekta"] = "generator_or_bay"

        suffix_part = f"|{meritev_suffix}" if meritev_suffix else ""

        out["component_id"] = f"G|{lokacija_od}|{napetost}|{tail}{suffix_part}"
        out["parse_status"] = "review"
        return out

    # 7) Plinski bloki PB
    if tail.startswith("PB"):
        out["tip_objekta"] = "gas_block"
        out["component_id"] = f"PB|{lokacija_od}|{napetost}|{tail}"
        out["parse_status"] = "ok"
        return out

    # 8) Bloki tipa BLOK4, BLOK5, BLOK6
    if tail.startswith("BLOK"):
        out["tip_objekta"] = "block"
        out["component_id"] = f"BLOK|{lokacija_od}|{napetost}|{tail}"
        out["parse_status"] = "ok"
        return out

    # 9) BHEE / posebna merilna mesta
    if lokacija_od.startswith("BHEE"):
        out["tip_objekta"] = "bess_or_metering"
        out["component_id"] = f"BHEE|{lokacija_od}|{napetost}|{tail}"
        out["parse_status"] = "review"
        return out

    # 10) Meritve / posebni odjemi
    # Primeri:
    # AJDOVSCINA110MERITVE2UB
    # RAVNEZEL5EPECP
    # RAVNEZEL5KPTEQ
    # RAVNEZEL5SPTE1P
    if (
        "MERITVE" in tail
        or tail.startswith("EPEC")
        or tail.startswith("KPTE")
        or tail.startswith("SPTE")
    ):
        out["tip_objekta"] = "metering_or_load"

        suffix_part = f"|{meritev_suffix}" if meritev_suffix else ""

        out["component_id"] = f"METER|{lokacija_od}|{napetost}|{tail}{suffix_part}"
        out["parse_status"] = "review"
        return out

    # 11) Vse ostalo: daljnovod / povezava
    lokacija_do, oznaka = split_line_tail(tail)

    out["tip_objekta"] = "line"
    out["lokacija_do"] = lokacija_do
    out["oznaka"] = oznaka
    out["component_id"] = f"LINE|{lokacija_od}|{napetost}|{lokacija_do}|{oznaka or ''}"
    out["parse_status"] = "ok"

    return out

# 1) METADATA REVIEW

def build_metadata_review():
    csv_files = sorted(INPUT_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"Ni CSV datotek v mapi: {INPUT_DIR}")

    unique_lfs = []

    for file in csv_files:
        print(f"Berem unikatne exid iz: {file.name}")

        lf = scan_csv_clean(file)
        cols = lf.collect_schema().names()

        if "exid" not in cols:
            raise KeyError(f"Datoteka nima stolpca exid: {file.name}. Stolpci: {cols}")

        unique_lfs.append(
            lf.select(
                pl.col("exid")
                .cast(pl.Utf8)
                .str.strip_chars()
                .str.to_uppercase()
                .alias("exid_key")
            )
            .drop_nulls()
            .unique()
        )

    unique_exids = pl.concat(unique_lfs).unique().collect()

    metadata_rows = [
        parse_exid(x)
        for x in unique_exids["exid_key"].to_list()
    ]

    metadata = pl.DataFrame(
    metadata_rows,
    infer_schema_length=None,
    schema_overrides={
        "exid_key": pl.Utf8,
        "exid_original": pl.Utf8,
        "lokacija_od": pl.Utf8,
        "napetost_kv": pl.Int64,
        "tip_objekta": pl.Utf8,
        "objekt": pl.Utf8,
        "lokacija_do": pl.Utf8,
        "oznaka": pl.Utf8,
        "meritev": pl.Utf8,
        "meritev_suffix": pl.Utf8,
        "component_id": pl.Utf8,
        "parse_status": pl.Utf8,
    }
)

    metadata = metadata.sort(["tip_objekta", "lokacija_od", "napetost_kv", "objekt", "meritev"])

    metadata.write_csv(METADATA_FILE)

    print("\nMetadata shranjen:")
    print(METADATA_FILE)

    print("\nTipi objektov:")
    print(metadata.group_by("tip_objekta").len().sort("len", descending=True))

    print("\nParse status:")
    print(metadata.group_by("parse_status").len().sort("len", descending=True))

def export_problem_exids():
    if not METADATA_FILE.exists():
        raise FileNotFoundError("Najprej zaženi MODE = 'metadata'.")

    metadata = pl.read_csv(METADATA_FILE, infer_schema_length=10000)

    problems = (
        metadata
        .filter(
            (pl.col("parse_status") != "ok") |
            (pl.col("tip_objekta") == "unknown")
        )
        .sort(["parse_status", "tip_objekta", "exid_key"])
    )

    out_file = OUT_DIR / "problem_exids.csv"
    problems.write_csv(out_file)

    print(f"Problematični exid shranjeni v:")
    print(out_file)

    print("\nPregled:")
    print(
        problems
        .group_by(["parse_status", "tip_objekta"])
        .len()
        .sort("len", descending=True)
    )

    print("\nPrvih 100 problematičnih exid:")
    print(problems.select(["exid_key", "tip_objekta", "parse_status"]).head(100))

# 2) NORMALIZACIJA V PARQUET

def optional_col(names, name, dtype):
    if name in names:
        return pl.col(name).cast(dtype, strict=False).alias(name)
    return pl.lit(None).cast(dtype).alias(name)

def parse_systime_expr(col_name: str = "systime"):
    s = (
        pl.col(col_name)
        .cast(pl.Utf8)
        .str.strip_chars()
        .str.replace_all(r"\s+", " ")
        .str.replace_all(r"\s*\.\s*", ".")
        .str.replace_all(r"\s*-\s*", "-")
        .str.replace_all(r"\s*/\s*", "/")
    )

    # Format 1: d.m.yyyy hh:mm[:ss]
    g_dot = s.str.extract_groups(
        r"^(?P<day>\d{1,2})\.(?P<month>\d{1,2})\.(?P<year>\d{4}) "
        r"(?P<hour>\d{1,2}):(?P<minute>\d{1,2})(?::(?P<second>\d{1,2}))?.*$"
    )

    dt_dot = pl.datetime(
        year=g_dot.struct.field("year").cast(pl.Int32, strict=False),
        month=g_dot.struct.field("month").cast(pl.Int32, strict=False),
        day=g_dot.struct.field("day").cast(pl.Int32, strict=False),
        hour=g_dot.struct.field("hour").cast(pl.Int32, strict=False),
        minute=g_dot.struct.field("minute").cast(pl.Int32, strict=False),
        second=pl.coalesce([
            g_dot.struct.field("second").cast(pl.Int32, strict=False),
            pl.lit(0)
        ])
    )

    # Format 2: yyyy-mm-dd hh:mm[:ss]
    g_iso = s.str.extract_groups(
        r"^(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2}) "
        r"(?P<hour>\d{1,2}):(?P<minute>\d{1,2})(?::(?P<second>\d{1,2}))?.*$"
    )

    dt_iso = pl.datetime(
        year=g_iso.struct.field("year").cast(pl.Int32, strict=False),
        month=g_iso.struct.field("month").cast(pl.Int32, strict=False),
        day=g_iso.struct.field("day").cast(pl.Int32, strict=False),
        hour=g_iso.struct.field("hour").cast(pl.Int32, strict=False),
        minute=g_iso.struct.field("minute").cast(pl.Int32, strict=False),
        second=pl.coalesce([
            g_iso.struct.field("second").cast(pl.Int32, strict=False),
            pl.lit(0)
        ])
    )

    return pl.coalesce([dt_dot, dt_iso])

def parse_value_expr(col_name: str = "value"):
    return (
        pl.col(col_name)
        .cast(pl.Utf8)
        .str.strip_chars()
        .str.replace_all(r"\s+", "")
        .str.replace_all(",", ".")
        .cast(pl.Float64, strict=False)
    )

def normalize_all_csv_to_parquet():
    if not METADATA_FILE.exists():
        raise FileNotFoundError("Najprej zaženi MODE = 'metadata'.")
    # počisti stare parquet datoteke, da ne mešaš starih in novih rezultatov
    for old_file in PARQUET_DIR.glob("*.parquet"):
        old_file.unlink()
    metadata_lf = (
        pl.scan_csv(METADATA_FILE, infer_schema_length=10000)
        .select([
            "exid_key",
            "lokacija_od",
            "napetost_kv",
            "tip_objekta",
            "objekt",
            "lokacija_do",
            "oznaka",
            "meritev",
            "meritev_suffix",
            "component_id",
            "parse_status",
        ])
    )

    csv_files = sorted(INPUT_DIR.glob("*.csv"))

    for file in csv_files:
        print(f"Normaliziram: {file.name}")

        lf = scan_csv_clean(file)
        cols = lf.collect_schema().names()

        required = ["exid", "systime", "value"]
        missing = [c for c in required if c not in cols]

        if missing:
            raise KeyError(
                f"Manjkajo stolpci {missing} v datoteki {file.name}. "
                f"Najdeni stolpci: {cols}"
            )

        clean = lf.select(
            pl.col("exid")
            .cast(pl.Utf8)
            .str.strip_chars()
            .str.to_uppercase()
            .alias("exid_key"),

            pl.col("exid")
            .cast(pl.Utf8)
            .str.strip_chars()
            .alias("exid_original"),

            parse_systime_expr("systime").alias("time"),

            parse_value_expr("value").alias("value"),

            optional_col(cols, "qst_no", pl.Int64),
            optional_col(cols, "timestamp", pl.Utf8),
            optional_col(cols, "calc_counter", pl.Int64),
        )
        # DEBUG PRED JOIN IN PRED FILTRI
        pre_filter_debug = clean.select(
            pl.len().alias("clean_rows"),
            pl.col("time").null_count().alias("time_nulls"),
            pl.col("value").null_count().alias("value_nulls"),
            pl.col("exid_key").null_count().alias("exid_nulls"),
        ).collect()

        joined_unfiltered = clean.join(metadata_lf, on="exid_key", how="left")

        post_join_debug = joined_unfiltered.select(
            pl.len().alias("joined_rows"),
            pl.col("time").null_count().alias("time_nulls"),
            pl.col("value").null_count().alias("value_nulls"),
            pl.col("component_id").null_count().alias("component_id_nulls"),
        ).collect()

        joined = (
            joined_unfiltered
            .filter(pl.col("time").is_not_null())
            .filter(pl.col("value").is_not_null())
        )

        after_filter_debug = joined.select(
            pl.len().alias("rows_after_filter"),
            pl.col("component_id").null_count().alias("component_id_nulls"),
        ).collect()


        if ONLY_QST_1:
            joined = joined.filter(pl.col("qst_no") == 1)

        out_file = PARQUET_DIR / f"{file.stem}.parquet"

        joined.sink_parquet(out_file)

        print(f"Shranjeno: {out_file}")


# 3) WIDE TABELE: P, Q, U, T PO OBJEKTIH

def make_wide_table(tip_objekta: str, out_name: str):
    lf = pl.scan_parquet(str(PARQUET_DIR / "*.parquet"))

    keys = [
        "time",
        "component_id",
        "tip_objekta",
        "lokacija_od",
        "lokacija_do",
        "napetost_kv",
        "objekt",
        "oznaka",
        "meritev_suffix",
    ]

    wide = (
        lf
        .filter(pl.col("tip_objekta") == tip_objekta)
        .filter(pl.col("qst_no") == 1)  # uporabi samo dobre meritve
        .group_by(keys)
        .agg(
            pl.col("value")
            .filter(pl.col("meritev") == "P")
            .first()
            .alias("P"),

            pl.col("value")
            .filter(pl.col("meritev") == "Q")
            .first()
            .alias("Q"),

            pl.col("value")
            .filter(pl.col("meritev") == "U")
            .first()
            .alias("U"),
        )
        .sort(["component_id", "time"])
    )

    out_file = OUT_DIR / out_name
    wide.sink_parquet(out_file)

    print(f"Wide tabela shranjena: {out_file}")

def safe_filename(name: str) -> str:
    name = str(name)
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = re.sub(r"\s+", "_", name)
    return name[:180]

def export_component_catalog():
    lf = pl.scan_parquet(str(PARQUET_DIR / "*.parquet"))

    catalog = (
        lf
        .select([
            "component_id",
            "tip_objekta",
            "lokacija_od",
            "lokacija_do",
            "napetost_kv",
            "objekt",
        ])
        .drop_nulls("component_id")
        .unique()
        .sort(["tip_objekta", "lokacija_od", "napetost_kv", "component_id"])
    )

    out_parquet = OUT_DIR / "component_catalog_clean.parquet"
    out_csv = OUT_DIR / "component_catalog_clean.csv"

    catalog.sink_parquet(out_parquet)
    catalog.collect().write_csv(out_csv)

    print("Clean component catalog shranjen:")
    print(out_parquet)
    print(out_csv)

def export_files_per_component():
    lf_base = pl.scan_parquet(str(PARQUET_DIR / "*.parquet"))

    if EXPORT_ONLY_QST_1:
        lf_base = lf_base.filter(pl.col("qst_no") == 1)

    # Počisti stare fajle
    for old_file in COMPONENT_EXPORT_DIR.glob("*.parquet"):
        old_file.unlink()

    components = (
        lf_base
        .select("component_id")
        .drop_nulls("component_id")
        .unique()
        .sort("component_id")
        .collect()
        .get_column("component_id")
        .to_list()
    )

    print(f"Število component_id za izvoz: {len(components)}")

    for i, cid in enumerate(components, start=1):
        file_id = cid.replace("|", "_")
        file_id = re.sub(r'[<>:"/\\?*]', "_", file_id)
        file_id = re.sub(r"\s+", "_", file_id)

        out_file = COMPONENT_EXPORT_DIR / f"{file_id}.parquet"

        component_wide = (
            lf_base
            .filter(pl.col("component_id") == cid)
            .group_by("time")
            .agg(
                pl.col("value").filter(pl.col("meritev") == "P").first().alias("P"),
                pl.col("value").filter(pl.col("meritev") == "Q").first().alias("Q"),
                pl.col("value").filter(pl.col("meritev") == "U").first().alias("U"),
            )
            .sort("time")
        )

        component_wide.sink_parquet(out_file)

        if i % 100 == 0:
            print(f"Izvoženo {i}/{len(components)}")

    print("Izvoz po component_id končan.")
    print(f"Mapa: {COMPONENT_EXPORT_DIR}")

# ZAGON

if __name__ == "__main__":

    if MODE == "metadata":
        build_metadata_review()

    elif MODE == "normalize":
        normalize_all_csv_to_parquet()
    elif MODE == "wide":
        make_wide_table("transformer", "transformers_wide.parquet")
        make_wide_table("line", "lines_wide.parquet")
        make_wide_table("generator", "generators_wide.parquet")
        make_wide_table("generator_or_bay", "generators_or_bays_wide.parquet")
        make_wide_table("gas_block", "gas_blocks_wide.parquet")
        make_wide_table("block", "blocks_wide.parquet")
        make_wide_table("metering_or_load", "metering_or_load_wide.parquet")
        make_wide_table("bess_or_metering", "bess_or_metering_wide.parquet")

    elif MODE == "component_catalog":
        export_component_catalog()

    elif MODE == "component_files":
        export_files_per_component()

    elif MODE == "problems":
        export_problem_exids()

    else:
        raise ValueError("MODE mora biti: metadata, problems, normalize, wide, component_catalog ali component_files")