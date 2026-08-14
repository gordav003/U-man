import argparse
import csv
import re
from pathlib import Path

import polars as pl

# ============================================================
# NASTAVITVE
# ============================================================

# Možni načini:
#   metadata        -> poišče vse EXID, ki se končajo na T, in izdela pregled
#   problems        -> izvozi TAP EXID, ki jih parser ni zanesljivo razdelil
#   normalize       -> normalizira vse TAP meritve iz vhodnih CSV v parquet
#   component_files -> izdela eno parquet datoteko za vsak TAP EXID
#   catalog         -> izdela katalog vseh TAP meritev
#   all             -> izvede metadata + problems + normalize + catalog + component_files
MODES = ("metadata", "problems", "normalize", "catalog", "component_files", "all")

# Če je True, se ohranijo samo meritve s qst_no == 1.
ONLY_QST_1 = True

# Če je True, se pred novim izvozom pobrišejo stare datoteke v izhodnih mapah.
CLEAN_OLD_OUTPUT = True


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Prepare SCADA transformer tap measurements for U-MAN analysis."
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing input CSV files.")
    parser.add_argument(
        "-o", "--output-dir", type=Path,
        help="Output directory (default: INPUT_DIR/urejeno/Uman_TAP_parquet).",
    )
    parser.add_argument(
        "--mode", choices=MODES, default="all",
        help="Processing step to run (default: all).",
    )
    args = parser.parse_args()
    args.input_dir = args.input_dir.expanduser().resolve()
    if args.output_dir is None:
        args.output_dir = args.input_dir / "urejeno" / "Uman_TAP_parquet"
    else:
        args.output_dir = args.output_dir.expanduser().resolve()
    return args


def configure_paths(input_dir: Path, output_dir: Path) -> None:
    global INPUT_DIR, OUT_DIR, METADATA_FILE, PROBLEMS_FILE
    global CATALOG_PARQUET, CATALOG_CSV, NORMALIZED_DIR, COMPONENT_DIR

    INPUT_DIR = input_dir
    OUT_DIR = output_dir
    if not INPUT_DIR.is_dir():
        raise NotADirectoryError(f"Vhodna mapa ne obstaja: {INPUT_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_FILE = OUT_DIR / "tap_metadata_review.csv"
    PROBLEMS_FILE = OUT_DIR / "tap_problem_exids.csv"
    CATALOG_PARQUET = OUT_DIR / "tap_catalog.parquet"
    CATALOG_CSV = OUT_DIR / "tap_catalog.csv"
    NORMALIZED_DIR = OUT_DIR / "tap_parquet_normalized"
    COMPONENT_DIR = OUT_DIR / "tap_component_files"
    NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)
    COMPONENT_DIR.mkdir(parents=True, exist_ok=True)

BASE_COLUMNS = [
    "exid",
    "systime",
    "value",
    "qst_no",
    "timestamp",
    "calc_counter",
]

VOLTAGES = ["400", "220", "110", "35", "21", "20", "18", "13", "11", "10", "6", "5"]
VOLTAGES = sorted(VOLTAGES, key=len, reverse=True)
VOLTAGE_RE = "|".join(VOLTAGES)

# Znani repi TAP identifikatorjev. Primeri:
#   AJDOVSCINATR1T -> AJDOVSCINA + TR1 + T
#   BERICEVOT211T  -> BERICEVO + T211 + T
#   CIRKOVCEL401T  -> CIRKOVCE + L401 + T
#   SEZANATRAT     -> SEZANA + TRA + T
TAP_OBJECT_RE = r"TR[A-Z0-9]*|T\d+[A-Z]*|L\d+[A-Z]*|OLTC[A-Z0-9]*|REG[A-Z0-9]*"


# ============================================================
# CSV BRANJE
# ============================================================

def read_first_nonempty_lines(path: Path, n: int = 50) -> list[str]:
    lines: list[str] = []

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
    """Izbere separator, ki daje najbolj stabilno število stolpcev."""
    lines = read_first_nonempty_lines(path, n=50)

    if not lines:
        raise ValueError(f"Prazna datoteka: {path}")

    candidates = ["\t", ";", "|", ","]
    best_sep = None
    best_score = None

    for sep in candidates:
        counts = [count_fields(line, sep) for line in lines]
        valid_counts = [count for count in counts if count >= 5]

        if not valid_counts:
            continue

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

    fields = next(csv.reader([lines[0]], delimiter=sep))
    fields_clean = [
        field.replace("\ufeff", "").strip().lower().replace(" ", "")
        for field in fields
    ]

    return (
        "exid" in fields_clean
        or "systime(utc+1)" in fields_clean
        or "value" in fields_clean
    )


def detect_num_columns(path: Path, sep: str, skip_rows: int = 0, n: int = 100) -> int:
    lines = read_first_nonempty_lines(path, n=n + skip_rows)

    if skip_rows:
        lines = lines[skip_rows:]

    if not lines:
        return len(BASE_COLUMNS)

    counts = [count_fields(line, sep) for line in lines]
    return max(max(counts), len(BASE_COLUMNS))


def make_column_names(n_cols: int) -> list[str]:
    names = BASE_COLUMNS.copy()

    for i in range(max(0, n_cols - len(BASE_COLUMNS))):
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

    return pl.scan_csv(
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


# ============================================================
# TAP PARSER
# ============================================================

def is_tap_exid(exid: str) -> bool:
    """TAP meritev je EXID, katerega zadnji znak je T."""
    return str(exid).strip().upper().endswith("T")


def parse_tap_exid(exid: str) -> dict:
    """
    Razčleni TAP EXID, vendar nikoli ne zavrže veljavne meritve, ki se konča na T.

    Pri neznanem formatu se ohrani celoten body in parse_status='review'.
    Tako se tudi neprepoznani TAP-i normalizirajo in izvozijo.
    """
    original = str(exid).strip()
    x = original.upper()

    out = {
        "exid_key": x,
        "exid_original": original,
        "tap_body": None,
        "lokacija": None,
        "napetost_kv": None,
        "objekt": None,
        "tip_objekta": "tap",
        "meritev": "TAP",
        "component_id": None,
        "parse_status": "unparsed",
        "parse_note": None,
    }

    if not x.endswith("T") or len(x) < 2:
        out["parse_note"] = "EXID se ne konča na T"
        return out

    body = x[:-1]
    out["tap_body"] = body

    # Component ID temelji na celotnem TAP body-ju. S tem se preprečijo
    # kolizije tudi pri neobičajnih ali še nepoznanih oznakah.
    out["component_id"] = f"TAP|{body}"

    object_match = re.match(
        rf"^(?P<prefix>.+?)(?P<objekt>{TAP_OBJECT_RE})$",
        body,
    )

    if object_match is None:
        out.update({
            "lokacija": body,
            "objekt": "TAP",
            "parse_status": "review",
            "parse_note": "Neznan format objekta; meritev je vseeno ohranjena",
        })
        return out

    prefix = object_match.group("prefix")
    objekt = object_match.group("objekt")

    # Opcijski napetostni nivo tik pred oznako objekta, npr. AJDOVSCINA110TR1T.
    voltage_match = re.match(
        rf"^(?P<lokacija>.+?)(?P<napetost>{VOLTAGE_RE})$",
        prefix,
    )

    if voltage_match is not None:
        lokacija = voltage_match.group("lokacija")
        napetost_kv = int(voltage_match.group("napetost"))
    else:
        lokacija = prefix
        napetost_kv = None

    out.update({
        "lokacija": lokacija,
        "napetost_kv": napetost_kv,
        "objekt": objekt,
        "parse_status": "ok",
        "parse_note": None,
    })

    return out


# ============================================================
# PRETVORBE PODATKOV
# ============================================================

def optional_col(names: list[str], name: str, dtype: pl.DataType) -> pl.Expr:
    if name in names:
        return pl.col(name).cast(dtype, strict=False).alias(name)

    return pl.lit(None).cast(dtype).alias(name)


def parse_systime_expr(col_name: str = "systime") -> pl.Expr:
    s = (
        pl.col(col_name)
        .cast(pl.Utf8)
        .str.strip_chars()
        .str.replace_all(r"\s+", " ")
        .str.replace_all(r"\s*\.\s*", ".")
        .str.replace_all(r"\s*-\s*", "-")
        .str.replace_all(r"\s*/\s*", "/")
    )

    # d.m.yyyy hh:mm[:ss]
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
            pl.lit(0),
        ]),
    )

    # yyyy-mm-dd hh:mm[:ss]
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
            pl.lit(0),
        ]),
    )

    return pl.coalesce([dt_dot, dt_iso])


def parse_tap_value_expr(col_name: str = "value") -> pl.Expr:
    """
    TAP ostane Float64, ker s tem ne izgubimo podatka, če se v viru pojavi
    decimalna ali posebna kodirana vrednost. Pri običajnem OLTC bo vrednost cela.
    """
    return (
        pl.col(col_name)
        .cast(pl.Utf8)
        .str.strip_chars()
        .str.replace_all(r"\s+", "")
        .str.replace_all(",", ".")
        .cast(pl.Float64, strict=False)
    )


def clean_directory(directory: Path) -> None:
    if not CLEAN_OLD_OUTPUT:
        return

    for old_file in directory.glob("*.parquet"):
        old_file.unlink()


# ============================================================
# 1) METADATA
# ============================================================

def build_tap_metadata() -> None:
    csv_files = sorted(INPUT_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"Ni CSV datotek v mapi: {INPUT_DIR}")

    unique_lfs: list[pl.LazyFrame] = []

    for file in csv_files:
        print(f"Iščem TAP EXID v: {file.name}")
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
            .filter(pl.col("exid_key").str.ends_with("T"))
            .drop_nulls()
            .unique()
        )

    unique_exids = (
        pl.concat(unique_lfs)
        .unique()
        .sort("exid_key")
        .collect()
    )

    if unique_exids.is_empty():
        raise RuntimeError("V vhodnih CSV datotekah ni bil najden noben EXID, ki se konča na T.")

    metadata_rows = [
        parse_tap_exid(exid)
        for exid in unique_exids.get_column("exid_key").to_list()
    ]

    metadata = pl.DataFrame(
        metadata_rows,
        infer_schema_length=None,
        schema_overrides={
            "exid_key": pl.Utf8,
            "exid_original": pl.Utf8,
            "tap_body": pl.Utf8,
            "lokacija": pl.Utf8,
            "napetost_kv": pl.Int64,
            "objekt": pl.Utf8,
            "tip_objekta": pl.Utf8,
            "meritev": pl.Utf8,
            "component_id": pl.Utf8,
            "parse_status": pl.Utf8,
            "parse_note": pl.Utf8,
        },
    ).sort(["parse_status", "lokacija", "objekt", "exid_key"])

    metadata.write_csv(METADATA_FILE)

    print()
    print(f"TAP metadata: {METADATA_FILE}")
    print(f"Število unikatnih TAP EXID: {metadata.height}")
    print(metadata.group_by("parse_status").len().sort("len", descending=True))


def export_problem_exids() -> None:
    if not METADATA_FILE.exists():
        raise FileNotFoundError("Najprej zaženi MODE = 'metadata' ali MODE = 'all'.")

    metadata = pl.read_csv(METADATA_FILE, infer_schema_length=10000)
    problems = (
        metadata
        .filter(pl.col("parse_status") != "ok")
        .sort(["parse_status", "exid_key"])
    )

    problems.write_csv(PROBLEMS_FILE)
    print(f"Problematični oziroma neznani TAP EXID: {PROBLEMS_FILE}")
    print(f"Število zapisov za pregled: {problems.height}")


# ============================================================
# 2) NORMALIZACIJA TAP MERITEV
# ============================================================

def normalize_tap_measurements() -> None:
    if not METADATA_FILE.exists():
        raise FileNotFoundError("Najprej zaženi MODE = 'metadata' ali MODE = 'all'.")

    clean_directory(NORMALIZED_DIR)

    metadata_lf = (
        pl.scan_csv(METADATA_FILE, infer_schema_length=10000)
        .select([
            "exid_key",
            "tap_body",
            "lokacija",
            "napetost_kv",
            "objekt",
            "tip_objekta",
            "meritev",
            "component_id",
            "parse_status",
        ])
    )

    csv_files = sorted(INPUT_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"Ni CSV datotek v mapi: {INPUT_DIR}")

    total_written = 0

    for file in csv_files:
        print(f"Normaliziram TAP: {file.name}")

        lf = scan_csv_clean(file)
        cols = lf.collect_schema().names()
        required = ["exid", "systime", "value"]
        missing = [column for column in required if column not in cols]

        if missing:
            raise KeyError(
                f"Manjkajo stolpci {missing} v datoteki {file.name}. "
                f"Najdeni stolpci: {cols}"
            )

        clean = (
            lf.select(
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
                parse_tap_value_expr("value").alias("TAP"),

                optional_col(cols, "qst_no", pl.Int64),
                optional_col(cols, "timestamp", pl.Utf8),
                optional_col(cols, "calc_counter", pl.Int64),
            )
            .filter(pl.col("exid_key").str.ends_with("T"))
            .filter(pl.col("time").is_not_null())
            .filter(pl.col("TAP").is_not_null())
        )

        joined = clean.join(metadata_lf, on="exid_key", how="left")

        if ONLY_QST_1:
            joined = joined.filter(pl.col("qst_no") == 1)

        # Izraz je True pri običajnih celoštevilskih pozicijah OLTC.
        joined = joined.with_columns(
            ((pl.col("TAP") - pl.col("TAP").round(0)).abs() < 1e-9)
            .alias("tap_is_integer")
        )

        out_file = NORMALIZED_DIR / f"{file.stem}_TAP.parquet"
        joined.sink_parquet(out_file)

        n_rows = pl.scan_parquet(out_file).select(pl.len()).collect().item()
        total_written += n_rows
        print(f"  -> {out_file.name} | vrstic: {n_rows}")

    print()
    print(f"Normalizacija TAP končana. Skupaj vrstic: {total_written}")
    print(f"Mapa: {NORMALIZED_DIR}")


# ============================================================
# 3) KATALOG IN DATOTEKE PO TAP KOMPONENTAH
# ============================================================

def export_tap_catalog() -> None:
    if not METADATA_FILE.exists():
        raise FileNotFoundError("Najprej zaženi MODE = 'metadata' ali MODE = 'all'.")

    catalog = (
        pl.read_csv(METADATA_FILE, infer_schema_length=10000)
        .select([
            "component_id",
            "exid_key",
            "tap_body",
            "lokacija",
            "napetost_kv",
            "objekt",
            "parse_status",
            "parse_note",
        ])
        .unique()
        .sort(["lokacija", "objekt", "exid_key"])
    )

    catalog.write_parquet(CATALOG_PARQUET)
    catalog.write_csv(CATALOG_CSV)

    print(f"TAP katalog parquet: {CATALOG_PARQUET}")
    print(f"TAP katalog CSV:     {CATALOG_CSV}")


def safe_filename(name: str) -> str:
    name = str(name)
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")[:180]


def export_tap_component_files() -> None:
    parquet_files = sorted(NORMALIZED_DIR.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(
            f"Ni normaliziranih TAP parquet datotek v: {NORMALIZED_DIR}. "
            "Najprej zaženi MODE = 'normalize' ali MODE = 'all'."
        )

    clean_directory(COMPONENT_DIR)

    lf = pl.scan_parquet(str(NORMALIZED_DIR / "*.parquet"))

    components = (
        lf.select("component_id")
        .drop_nulls("component_id")
        .unique()
        .sort("component_id")
        .collect()
        .get_column("component_id")
        .to_list()
    )

    print(f"Število TAP komponent za izvoz: {len(components)}")

    for i, component_id in enumerate(components, start=1):
        tap_body = component_id.split("|", maxsplit=1)[-1]
        out_file = COMPONENT_DIR / f"TAP_{safe_filename(tap_body)}.parquet"

        # Pri podvojenih zapisih v istem času se ohrani zadnja vrednost,
        # dodatna stolpca pa pokažeta, ali je bil vir konsistenten.
        component_wide = (
            lf.filter(pl.col("component_id") == component_id)
            .sort(["time", "timestamp", "calc_counter"], nulls_last=True)
            .group_by("time")
            .agg(
                pl.col("TAP").last().alias("TAP"),
                pl.col("qst_no").min().alias("qst_no"),
                pl.len().alias("n_source_rows"),
                pl.col("TAP").n_unique().alias("n_unique_tap"),
                pl.col("tap_is_integer").all().alias("tap_is_integer"),
                pl.col("exid_key").first().alias("exid"),
                pl.col("lokacija").first().alias("lokacija"),
                pl.col("napetost_kv").first().alias("napetost_kv"),
                pl.col("objekt").first().alias("objekt"),
                pl.col("parse_status").first().alias("parse_status"),
            )
            .sort("time")
            .select([
                "time",
                "TAP",
                "qst_no",
                "n_source_rows",
                "n_unique_tap",
                "tap_is_integer",
                "exid",
                "lokacija",
                "napetost_kv",
                "objekt",
                "parse_status",
            ])
        )

        component_wide.sink_parquet(out_file)

        if i % 100 == 0 or i == len(components):
            print(f"Izvoženo {i}/{len(components)}")

    print("Izvoz TAP datotek po komponentah končan.")
    print(f"Mapa: {COMPONENT_DIR}")


# ============================================================
# ZAGON
# ============================================================

def main() -> None:
    args = parse_arguments()
    configure_paths(args.input_dir, args.output_dir)

    if args.mode == "metadata":
        build_tap_metadata()

    elif args.mode == "problems":
        export_problem_exids()

    elif args.mode == "normalize":
        normalize_tap_measurements()

    elif args.mode == "catalog":
        export_tap_catalog()

    elif args.mode == "component_files":
        export_tap_component_files()

    elif args.mode == "all":
        build_tap_metadata()
        export_problem_exids()
        normalize_tap_measurements()
        export_tap_catalog()
        export_tap_component_files()


if __name__ == "__main__":
    main()
