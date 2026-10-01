"""Prepare NHTSA owner complaints for modelling: one row per complaint (ODINO).

* Reads all 51 documented fields with QUOTE_NONE (the flat file is unquoted).
* Keeps vehicle complaints (PROD_TYPE == 'V'); tyre, equipment and child-seat products are
  separate product types with their own taxonomies.
* Harmonises COMPDESC level-1 names across taxonomy revisions into component groups
  (outputs/tables/complaint_component_mapping.csv).
* U1 target: multi-hot component groups. U2 target: complaint reports a crash, fire,
  injury or death (structured flags, never derived from the text).
* Removes NHTSA editorial artefacts from narratives, audits exact duplicates and assigns
  a chronological split: train <= 2021, validation 2022-2023, test 2024+.
"""
import csv
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feasibility_audit import COMPLAINT_COLUMNS
from project_paths import PROCESSED_DIR, TABLE_DIR, complaint_file

GROUPS = {
    "ENGINE": ["ENGINE", "ENGINE AND ENGINE COOLING"],
    "POWER TRAIN": ["POWER TRAIN"],
    "ELECTRICAL": ["ELECTRICAL SYSTEM", "COMMUNICATION", "COMMUNICATIONS"],
    "BRAKES": ["SERVICE BRAKES", "SERVICE BRAKES, HYDRAULIC", "SERVICE BRAKES, AIR",
               "SERVICE BRAKES, ELECTRIC", "PARKING BRAKE"],
    "STEERING": ["STEERING"],
    "SUSPENSION": ["SUSPENSION"],
    "AIR BAGS": ["AIR BAGS"],
    "SEAT BELTS": ["SEAT BELTS"],
    "SEATS": ["SEATS"],
    "STRUCTURE": ["STRUCTURE"],
    "LATCHES/LOCKS": ["LATCHES/LOCKS/LINKAGES"],
    "SPEED CONTROL": ["VEHICLE SPEED CONTROL"],
    "TIRES": ["TIRES"],
    "WHEELS": ["WHEELS"],
    "LIGHTING": ["EXTERIOR LIGHTING", "INTERIOR LIGHTING"],
    "FUEL/PROPULSION": ["FUEL/PROPULSION SYSTEM", "FUEL SYSTEM, GASOLINE", "FUEL SYSTEM, DIESEL",
                        "FUEL SYSTEM, OTHER", "HYBRID PROPULSION SYSTEM"],
    "VISIBILITY": ["VISIBILITY", "VISIBILITY/WIPER"],
    "DRIVER ASSISTANCE": ["FORWARD COLLISION AVOIDANCE", "LANE DEPARTURE", "BACK OVER PREVENTION",
                          "ELECTRONIC STABILITY CONTROL (ESC)", "ELECTRONIC STABILITY CONTROL",
                          "TRACTION CONTROL SYSTEM"],
    "EQUIPMENT": ["EQUIPMENT", "EQUIPMENT ADAPTIVE/MOBILITY", "TRAILER HITCHES", "TRAILER HARDWARE"],
}
TO_GROUP = {name: g for g, names in GROUPS.items() for name in names}
SPLIT_BOUNDS = {"train": (0, 2021), "validation": (2022, 2023), "test": (2024, 9999)}
# Editorial artefacts: editor initials such as "*TR", "*JB" and FOIA redaction notices.
ARTEFACTS = re.compile(r"\*[A-Z]{2,3}\b|INFORMATION REDACTED PURSUANT TO THE FREEDOM OF INFORMATION ACT \(FOIA\),? 5 U\.S\.C\. 552\(B\)\(6\)\.?|\[XXX\]")


def l1(compdesc: pd.Series) -> pd.Series:
    return compdesc.str.split(":").str[0].str.strip().str.upper()


def main() -> None:
    raw = pd.read_csv(complaint_file(), sep="\t", names=COMPLAINT_COLUMNS, dtype=str, encoding="latin-1",
                      quoting=csv.QUOTE_NONE, keep_default_na=False)
    n_raw = len(raw)
    df = raw[raw["PROD_TYPE"] == "V"].copy()
    df["component_l1"] = l1(df["COMPDESC"])
    df["group"] = df["component_l1"].map(TO_GROUP)

    mapping = (df.groupby("component_l1").size().rename("rows").reset_index()
               .assign(group=lambda d: d["component_l1"].map(TO_GROUP).fillna("(not a component: excluded from U1 labels)")))
    mapping.sort_values("rows", ascending=False).to_csv(TABLE_DIR / "complaint_component_mapping.csv", index=False)

    flag = lambda s: s.eq("Y").any()
    pos = lambda s: (pd.to_numeric(s, errors="coerce").fillna(0) > 0).any()
    df["_len"] = df["CDESCR"].str.len()
    df = df.sort_values(["ODINO", "_len"], ascending=[True, False])
    g = df.groupby("ODINO", sort=False)
    out = g.agg(narrative=("CDESCR", "first"), LDATE=("LDATE", "first"), DATEA=("DATEA", "first"),
                make=("MAKETXT", "first"), model=("MODELTXT", "first"), model_year=("YEARTXT", "first"),
                state=("STATE", "first"), crash=("CRASH", flag), fire=("FIRE", flag),
                injured=("INJURED", pos), deaths=("DEATHS", pos), n_rows=("CDESCR", "size"))
    groups = df.dropna(subset=["group"]).groupby("ODINO")["group"].agg(lambda s: sorted(set(s)))
    out["components"] = groups.reindex(out.index)
    out["components"] = out["components"].apply(lambda v: v if isinstance(v, list) else [])
    for name in GROUPS:
        out[f"c__{name}"] = out["components"].apply(lambda v, n=name: n in v).astype("int8")
    out["y_serious"] = out[["crash", "fire", "injured", "deaths"]].any(axis=1).astype("int8")

    out["text"] = (out["narrative"].str.replace(ARTEFACTS, " ", regex=True)
                   .str.replace(r"\s+", " ", regex=True).str.strip())
    out["date"] = pd.to_datetime(out["LDATE"], format="%Y%m%d", errors="coerce")
    out["year"] = out["date"].dt.year
    out = out[out["text"].str.len() >= 20].dropna(subset=["date"])
    out["split"] = "train"
    for split, (lo, hi) in SPLIT_BOUNDS.items():
        out.loc[out["year"].between(lo, hi), "split"] = split

    # Exact-duplicate audit on normalised text: keep the earliest occurrence only, so no
    # validation/test narrative is a verbatim copy of a training narrative.
    norm = out["text"].str.upper().str.replace(r"[^A-Z0-9 ]", "", regex=True)
    out = out.assign(_norm=norm).sort_values("date")
    dup = out["_norm"].duplicated(keep="first")
    dup_report = out[dup].groupby("split").size().rename("exact_duplicates_removed")
    out = out[~dup].drop(columns="_norm").reset_index()

    summary = out.groupby("split").agg(complaints=("ODINO", "size"), first=("date", "min"), last=("date", "max"),
                                       serious_rate=("y_serious", "mean"),
                                       with_component=("components", lambda s: (s.str.len() > 0).mean()),
                                       multi_component=("components", lambda s: (s.str.len() > 1).mean()))
    summary = summary.join(dup_report)
    summary.to_csv(TABLE_DIR / "complaint_split_summary.csv")
    label_dist = out.groupby("split")[[f"c__{n}" for n in GROUPS]].mean().T
    label_dist.to_csv(TABLE_DIR / "complaint_label_distribution.csv")

    keep = ["ODINO", "date", "year", "split", "make", "model", "model_year", "state", "text", "components",
            "crash", "fire", "injured", "deaths", "y_serious", "n_rows"] + [f"c__{n}" for n in GROUPS]
    out[keep].to_parquet(PROCESSED_DIR / "complaints_model.parquet", index=False)
    print(f"raw rows {n_raw:,} -> vehicle rows {len(df):,} -> complaints {len(out):,}")
    print(summary.to_string())
    print(label_dist.round(3).to_string())


if __name__ == "__main__":
    main()
