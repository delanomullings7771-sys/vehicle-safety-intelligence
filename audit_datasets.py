"""
Data audit for the capstone datasets (CRISP-DM: Data Understanding).

Checks that the NHTSA CRSS 2020-2024 files and the NHTSA Vehicle Owner
Complaints file load correctly, link correctly, and have usable targets.

Usage:
    1. Save this file inside the project folder (PROJECT_ROOT below).
    2. pip install pandas tabulate
    3. python audit_datasets.py
    4. Share the generated audit_report.md.
The script finds the CRSS year folders and the complaints file by itself.
"""

import csv
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------- settings
# The project folder. Everything below it is searched automatically.
PROJECT_ROOT = Path(r"C:\Users\delan\Downloads\NHTSA_Road_Vehicle_Safety_Capstone_Last_Lap")
CRSS_YEARS = [2020, 2021, 2022, 2023, 2024]

# Leave as None to auto-detect the complaints file (e.g. FLAT_CMPL.txt).
COMPLAINTS_FILE = None

# Paste ALL field names, in order, from the official complaint layout file.
# Only the fields named below are used by this audit; the rest are counted.
COMPLAINT_COLUMNS = [
    # e.g. "CMPLID", "ODINO", "MFR_NAME", "MAKETXT", "MODELTXT", "YEARTXT",
    #      "CRASH", "FAILDATE", "FIRE", "INJURED", "DEATHS", "COMPDESC", ...
]
RECEIVED_DATE_FIELD = "DATEA"   # date NHTSA received the complaint (YYYYMMDD)
REPORT_FILE = PROJECT_ROOT / "audit_report.md"
ENCODING = "latin-1"            # NHTSA files are not always valid UTF-8

report = []


def out(line=""):
    print(line)
    report.append(line)


def find_file(folder, name):
    """Case-insensitive recursive search for a file inside a year folder."""
    for p in folder.rglob("*"):
        if p.is_file() and p.name.lower() == name.lower():
            return p
    return None


def year_folder(year):
    """Folder holding ACCIDENT.CSV whose path mentions the year."""
    for acc in PROJECT_ROOT.rglob("*"):
        if acc.is_file() and acc.name.lower() == "accident.csv" and str(year) in str(acc.parent):
            return acc.parent
    return None


def find_complaints_file():
    candidates = [p for p in PROJECT_ROOT.rglob("*")
                  if p.is_file() and p.suffix.lower() == ".txt" and "cmpl" in p.name.lower()]
    return max(candidates, key=lambda p: p.stat().st_size) if candidates else None


def inventory():
    out("# Folder inventory\n")
    out(f"Root: {PROJECT_ROOT}\n")
    if not PROJECT_ROOT.exists():
        out("- ROOT FOLDER NOT FOUND. Check PROJECT_ROOT.")
        return False
    for p in sorted(PROJECT_ROOT.rglob("*")):
        if p.is_file():
            out(f"- {p.relative_to(PROJECT_ROOT)}  ({p.stat().st_size / 1e6:.1f} MB)")
    out()
    return True


def check_unique(df, keys, label):
    missing = [k for k in keys if k not in df.columns]
    if missing:
        out(f"- {label}: MISSING key columns {missing}")
        return
    dups = df.duplicated(subset=keys).sum()
    out(f"- {label} key {keys} duplicates: {dups:,}" + (" OK" if dups == 0 else " PROBLEM"))


# ---------------------------------------------------------------- CRSS
def audit_crss():
    out("# CRSS audit\n")
    columns_by_table = {"ACCIDENT": {}, "VEHICLE": {}, "PERSON": {}}
    target_rows = []

    for year in CRSS_YEARS:
        out(f"## {year}")
        folder = year_folder(year)
        if folder is None:
            out(f"- No ACCIDENT.CSV found for {year} under {PROJECT_ROOT}\n")
            continue

        tables = {}
        for name in ["ACCIDENT", "VEHICLE", "PERSON"]:
            path = find_file(folder, f"{name}.CSV")
            if path is None:
                out(f"- {name}.CSV not found")
                continue
            df = pd.read_csv(path, encoding=ENCODING, low_memory=False)
            tables[name] = df
            columns_by_table[name][year] = set(df.columns)
            out(f"- {name}: {len(df):,} rows, {df.shape[1]} columns, "
                f"{path.stat().st_size / 1e6:.1f} MB")

        vpic = find_file(folder, "vPICDecode.csv")
        out(f"- vPICDecode present: {'yes' if vpic else 'no'}")
        total_mb = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file()) / 1e6
        out(f"- All files in folder: {total_mb:.1f} MB")

        if "ACCIDENT" in tables:
            check_unique(tables["ACCIDENT"], ["CASENUM"], "ACCIDENT")
        if "VEHICLE" in tables:
            check_unique(tables["VEHICLE"], ["CASENUM", "VEH_NO"], "VEHICLE")
        if "PERSON" in tables:
            check_unique(tables["PERSON"], ["CASENUM", "VEH_NO", "PER_NO"], "PERSON")

        # Orphans: child records whose crash is missing from ACCIDENT
        if "ACCIDENT" in tables:
            crashes = set(tables["ACCIDENT"]["CASENUM"])
            for child in ["VEHICLE", "PERSON"]:
                if child in tables:
                    orphans = (~tables[child]["CASENUM"].isin(crashes)).sum()
                    out(f"- {child} rows with no matching crash: {orphans:,}")

        # Target distribution (raw and weighted)
        acc = tables.get("ACCIDENT")
        if acc is not None:
            for sev_col in ["MAX_SEV", "MAXSEV_IM"]:
                if sev_col not in acc.columns:
                    out(f"- {sev_col}: not present")
                    continue
                counts = acc[sev_col].value_counts().sort_index()
                out(f"- {sev_col} raw counts: {counts.to_dict()}")
                if "WEIGHT" in acc.columns:
                    w = acc.groupby(sev_col)["WEIGHT"].sum().round(0).sort_index()
                    out(f"- {sev_col} weighted totals: {w.to_dict()}")
                target_rows.append({"year": year, "field": sev_col,
                                    **{f"code_{k}": v for k, v in counts.items()}})
        out()

    # Column consistency across years
    out("## Column consistency across years")
    for name, by_year in columns_by_table.items():
        if not by_year:
            continue
        all_cols = set.union(*by_year.values())
        common = set.intersection(*by_year.values())
        out(f"- {name}: {len(common)} columns in every year, "
            f"{len(all_cols - common)} appear in only some years")
        for col in sorted(all_cols - common):
            years = [y for y, cols in by_year.items() if col in cols]
            out(f"  - {col}: {years}")
    out()

    if target_rows:
        out("## Severity code counts by year (check against the manual)")
        out(pd.DataFrame(target_rows).fillna(0).to_markdown(index=False))
        out("\nCodes must be interpreted with the annual Coding and Validation "
            "Manual before defining injury/fatal vs property-damage-only.\n")


# ---------------------------------------------------------------- complaints
def audit_complaints():
    global COMPLAINTS_FILE
    out("# Complaints audit\n")
    COMPLAINTS_FILE = Path(COMPLAINTS_FILE) if COMPLAINTS_FILE else find_complaints_file()
    if COMPLAINTS_FILE is None or not COMPLAINTS_FILE.exists():
        out("- Complaints .txt file not found (expected a name containing 'CMPL').")
        return
    out(f"- Using: {COMPLAINTS_FILE}")
    if not COMPLAINT_COLUMNS:
        with open(COMPLAINTS_FILE, encoding=ENCODING) as f:
            first = f.readline().rstrip("\r\n").split("\t")
        out(f"- COMPLAINT_COLUMNS is empty. First line has {len(first)} fields.")
        out("- Paste the field names from the layout file to run the full check.")
        return

    out(f"- File size: {COMPLAINTS_FILE.stat().st_size / 1e6:.1f} MB")
    n_expected = len(COMPLAINT_COLUMNS)

    # Count lines whose field count differs from the layout (tabs inside text etc.)
    bad_lines, total_lines = 0, 0
    with open(COMPLAINTS_FILE, encoding=ENCODING, newline="") as f:
        for line in f:
            total_lines += 1
            if line.rstrip("\r\n").count("\t") + 1 != n_expected:
                bad_lines += 1
    out(f"- Lines: {total_lines:,}; lines with wrong field count: {bad_lines:,}")

    reader = pd.read_csv(
        COMPLAINTS_FILE, sep="\t", header=None, names=COMPLAINT_COLUMNS,
        dtype=str, encoding=ENCODING, quoting=csv.QUOTE_NONE,
        on_bad_lines="skip", chunksize=200_000,
    )

    rows = 0
    per_year = pd.Series(dtype="int64")
    comp_top = pd.Series(dtype="int64")
    missing_text = missing_comp = 0
    rows_per_odino = pd.Series(dtype="int64")

    for chunk in reader:
        rows += len(chunk)
        missing_text += chunk["CDESCR"].isna().sum()
        missing_comp += chunk["COMPDESC"].isna().sum()

        year = chunk[RECEIVED_DATE_FIELD].str[:4]
        per_year = per_year.add(year.value_counts(), fill_value=0)

        recent = chunk[year.between("2020", "2025")]
        top = recent["COMPDESC"].str.split(":").str[0].str.strip()
        comp_top = comp_top.add(top.value_counts(), fill_value=0)
        rows_per_odino = rows_per_odino.add(recent["ODINO"].value_counts(), fill_value=0)

    out(f"- Rows parsed: {rows:,}")
    out(f"- Missing narratives (CDESCR): {missing_text:,}")
    out(f"- Missing component (COMPDESC): {missing_comp:,}\n")

    out("## Rows by year received")
    out(per_year.sort_index().astype(int).to_frame("rows").to_markdown())

    out("\n## 2020-2025: complaints vs rows")
    out(f"- Distinct complaints (ODINO): {len(rows_per_odino):,}")
    out(f"- Rows: {int(rows_per_odino.sum()):,}")
    out(f"- Complaints with more than one component row: "
        f"{int((rows_per_odino > 1).sum()):,}")

    out("\n## 2020-2025: top-level component categories (row counts)")
    out(comp_top.sort_values(ascending=False).astype(int).to_frame("rows").to_markdown())


if __name__ == "__main__":
    if not inventory():
        REPORT_FILE.write_text("\n".join(report), encoding="utf-8")
        sys.exit(1)
    try:
        audit_crss()
    except FileNotFoundError as e:
        out(f"CRSS audit failed: {e}")
    audit_complaints()
    REPORT_FILE.write_text("\n".join(report), encoding="utf-8")
    print(f"\nReport written to {REPORT_FILE.resolve()}", file=sys.stderr)
