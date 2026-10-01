"""Read-only inventory of the raw NHTSA datasets.

This script never modifies files under data/raw. It creates compact audit outputs
under outputs/tables for CRISP-DM Data Understanding.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from project_paths import CRSS_YEARS, TABLE_DIR, complaint_file, crss_extracted_dir


COMPLAINT_COLUMNS = [
    "CMPLID", "ODINO", "MFR_NAME", "MAKETXT", "MODELTXT", "YEARTXT",
    "CRASH", "FAILDATE", "FIRE", "INJURED", "DEATHS", "COMPDESC",
    "CITY", "STATE", "VIN", "DATEA", "LDATE", "MILES", "OCCURENCES",
    "CDESCR", "CMPL_TYPE", "POLICE_RPT_YN", "PURCH_DT", "ORIG_OWNER_YN",
    "ANTI_BRAKES_YN", "CRUISE_CONT_YN", "NUM_CYLS", "DRIVE_TRAIN",
    "FUEL_SYS", "FUEL_TYPE", "TRANS_TYPE", "VEH_SPEED", "DOT", "TIRE_SIZE",
    "LOC_OF_TIRE", "TIRE_FAIL_TYPE", "ORIG_EQUIP_YN", "MANUF_DT",
    "SEAT_TYPE", "RESTRAINT_TYPE", "DEALER_NAME", "DEALER_TEL", "DEALER_CITY",
    "DEALER_STATE", "DEALER_ZIP", "PROD_TYPE", "REPAIRED_YN", "MEDICAL_ATTN",
    "VEHICLES_TOWED_YN", "STATE_OF_INCIDENT", "VEHICLE_OPERATOR",
]


def sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def audit_crss() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for year in CRSS_YEARS:
        source_dir = crss_extracted_dir(year)
        csv_files = sorted(source_dir.glob("*.csv"), key=lambda p: p.name.lower())
        if len(csv_files) != 28:
            raise ValueError(f"Expected 28 CSV files for {year}; found {len(csv_files)}")
        for path in csv_files:
            with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
                reader = csv.reader(stream)
                header = next(reader)
                row_count = sum(1 for _ in reader)
            rows.append(
                {
                    "year": year,
                    "table": path.stem.lower(),
                    "rows": row_count,
                    "columns": len(header),
                    "size_bytes": path.stat().st_size,
                    "header": "|".join(header),
                    "sha256": sha256(path),
                }
            )
    return rows


def audit_complaints() -> dict[str, object]:
    path = complaint_file()
    width_counts: Counter[int] = Counter()
    blank_narratives = 0
    blank_components = 0
    unique_odino: set[str] = set()
    total = 0
    with path.open("r", encoding="latin-1", errors="replace", newline="") as stream:
        # The NHTSA flat file is unquoted; narratives contain literal '"' characters,
        # so default CSV quoting would merge neighbouring records.
        reader = csv.reader(stream, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            total += 1
            width_counts[len(row)] += 1
            if len(row) >= 20 and not row[19].strip():
                blank_narratives += 1
            if len(row) >= 12 and not row[11].strip():
                blank_components += 1
            if len(row) >= 2:
                unique_odino.add(row[1].strip())
    return {
        "file": str(path),
        "rows": total,
        "documented_columns": len(COMPLAINT_COLUMNS),
        "observed_row_widths": dict(sorted(width_counts.items())),
        "blank_narratives": blank_narratives,
        "blank_components": blank_components,
        "unique_odino": len(unique_odino),
        "repeated_component_rows": total - len(unique_odino),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def write_csv(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    print("Auditing all 140 CRSS CSV files...")
    crss = audit_crss()
    write_csv(TABLE_DIR / "crss_raw_file_inventory.csv", crss)
    print("Streaming the complete owner-complaint file...")
    complaints = audit_complaints()
    with (TABLE_DIR / "complaint_raw_file_audit.json").open("w", encoding="utf-8") as stream:
        json.dump(complaints, stream, indent=2)
    summary = {
        "crss_years": list(CRSS_YEARS),
        "crss_csv_files": len(crss),
        "crss_tables_per_year": Counter(row["year"] for row in crss),
        "complaints": complaints,
    }
    with (TABLE_DIR / "raw_data_audit_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
