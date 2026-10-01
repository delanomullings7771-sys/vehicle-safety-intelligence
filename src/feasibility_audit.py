"""Measure whether the raw data can support every planned model and the dataset linkage.

Outputs are written to outputs/tables/feasibility_*.csv|json. Raw data is read only.
"""
import csv
import json
import re
import sys

import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from project_paths import CRSS_YEARS, TABLE_DIR, complaint_file, crss_extracted_dir

COMPLAINT_COLUMNS = [
    "CMPLID", "ODINO", "MFR_NAME", "MAKETXT", "MODELTXT", "YEARTXT", "CRASH", "FAILDATE", "FIRE",
    "INJURED", "DEATHS", "COMPDESC", "CITY", "STATE", "VIN", "DATEA", "LDATE", "MILES", "OCCURENCES",
    "CDESCR", "CMPL_TYPE", "POLICE_RPT_YN", "PURCH_DT", "ORIG_OWNER_YN", "ANTI_BRAKES_YN",
    "CRUISE_CONT_YN", "NUM_CYLS", "DRIVE_TRAIN", "FUEL_SYS", "FUEL_TYPE", "TRANS_TYPE", "VEH_SPEED",
    "DOT", "TIRE_SIZE", "LOC_OF_TIRE", "TIRE_FAIL_TYPE", "ORIG_EQUIP_YN", "MANUF_DT", "SEAT_TYPE",
    "RESTRAINT_TYPE", "DEALER_NAME", "DEALER_TEL", "DEALER_CITY", "DEALER_STATE", "DEALER_ZIP",
    "PROD_TYPE", "REPAIRED_YN", "MEDICAL_ATTN", "VEHICLES_TOWED_YN", "STATE_OF_INCIDENT",
    "VEHICLE_OPERATOR",
]
USE = ["ODINO", "MAKETXT", "MODELTXT", "YEARTXT", "CRASH", "FIRE", "INJURED", "DEATHS",
       "COMPDESC", "LDATE", "DATEA", "CDESCR", "PROD_TYPE"]


def norm(s: pd.Series) -> pd.Series:
    return s.astype("string").str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)


def load_complaints() -> pd.DataFrame:
    rows = []
    with open(complaint_file(), encoding="latin-1", newline="") as fh:
        for line in fh:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) != 51:
                continue
            rows.append([parts[COMPLAINT_COLUMNS.index(c)] for c in USE])
    df = pd.DataFrame(rows, columns=USE)
    df["CDESCR_LEN"] = df["CDESCR"].str.split().str.len()
    df = df.drop(columns="CDESCR")
    df["component_l1"] = df["COMPDESC"].str.split(":").str[0].str.strip()
    df["year_received"] = pd.to_numeric(df["LDATE"].str[:4], errors="coerce")
    return df


def complaint_summary(df: pd.DataFrame) -> dict:
    grp = df.groupby("ODINO").agg(
        crash=("CRASH", lambda s: (s == "Y").any()),
        fire=("FIRE", lambda s: (s == "Y").any()),
        injured=("INJURED", lambda s: (pd.to_numeric(s, errors="coerce").fillna(0) > 0).any()),
        deaths=("DEATHS", lambda s: (pd.to_numeric(s, errors="coerce").fillna(0) > 0).any()),
        n_components=("component_l1", "nunique"),
        year=("year_received", "min"),
        words=("CDESCR_LEN", "max"),
    )
    grp["serious"] = grp[["crash", "fire", "injured", "deaths"]].any(axis=1)
    by_year = grp.groupby("year").agg(complaints=("serious", "size"), serious_rate=("serious", "mean"))
    by_year.to_csv(TABLE_DIR / "feasibility_complaints_by_year.csv")
    comp = df.drop_duplicates(["ODINO", "component_l1"])["component_l1"].value_counts()
    comp.to_csv(TABLE_DIR / "feasibility_complaint_components.csv")
    return {
        "rows_51_fields": int(len(df)),
        "unique_complaints_ODINO": int(len(grp)),
        "prod_type_counts": df["PROD_TYPE"].value_counts().head(6).to_dict(),
        "year_range": [int(grp["year"].min()), int(grp["year"].max())],
        "complaints_2015_onward": int((grp["year"] >= 2015).sum()),
        "label_rates_per_complaint": {k: round(float(grp[k].mean()), 4)
                                      for k in ["crash", "fire", "injured", "deaths", "serious"]},
        "serious_count": int(grp["serious"].sum()),
        "share_multi_component": round(float((grp["n_components"] > 1).mean()), 4),
        "narrative_words_median": float(grp["words"].median()),
        "distinct_component_l1": int(comp.size),
        "component_l1_with_1000plus": int((comp >= 1000).sum()),
    }


def crss_summary() -> tuple[dict, pd.DataFrame]:
    out, vehicles = {}, []
    for y in CRSS_YEARS:
        d = crss_extracted_dir(y)
        v = pd.read_csv(d / "vehicle.csv", encoding="latin-1", low_memory=False,
                        usecols=["CASENUM", "VEH_NO", "MOD_YEAR", "VPICMAKENAME", "VPICMODELNAME", "MAKENAME"])
        f = pd.read_csv(d / "factor.csv", encoding="latin-1", usecols=["CASENUM", "VEH_NO", "VEHICLECCNAME"])
        a = pd.read_csv(d / "accident.csv", encoding="latin-1", usecols=["CASENUM", "MAX_SEV"])
        defect = f[~f["VEHICLECCNAME"].isin(["None Noted", "Reported as Unknown"])]
        v = v.merge(a, on="CASENUM", how="left")
        v["year"] = y
        vehicles.append(v)
        inj = a.set_index("CASENUM")["MAX_SEV"].isin([1, 2, 3, 4])
        out[y] = {
            "crashes": len(a),
            "vehicles": len(v),
            "vehicles_with_defect_factor": int(defect[["CASENUM", "VEH_NO"]].drop_duplicates().shape[0]),
            "defect_factor_injury_crash_rate": round(float(inj.reindex(defect["CASENUM"].unique()).mean()), 4),
            "all_crash_injury_rate": round(float(inj.mean()), 4),
            "vehicles_with_decoded_make_model": int(v["VPICMODELNAME"].notna().sum()),
        }
        dist = defect["VEHICLECCNAME"].value_counts()
        dist.to_csv(TABLE_DIR / f"feasibility_crss_defect_factors_{y}.csv")
    return out, pd.concat(vehicles)


def linkage(cmp: pd.DataFrame, veh: pd.DataFrame) -> dict:
    c = cmp.assign(mk=norm(cmp["MAKETXT"]), md=norm(cmp["MODELTXT"]),
                   my=pd.to_numeric(cmp["YEARTXT"], errors="coerce"))
    c = c[c["year_received"] >= 2015]
    keys_c = set(zip(c["mk"], c["md"], c["my"]))
    v = veh.assign(mk=norm(veh["VPICMAKENAME"]), md=norm(veh["VPICMODELNAME"]),
                   my=pd.to_numeric(veh["MOD_YEAR"], errors="coerce"))
    v = v[v["my"].between(1990, 2026)]
    v["key"] = list(zip(v["mk"], v["md"], v["my"]))
    v["matched"] = v["key"].isin(keys_c)
    return {
        "crss_vehicles_with_valid_model_year": int(len(v)),
        "crss_vehicle_share_matching_complaint_make_model_year": round(float(v["matched"].mean()), 4),
        "distinct_make_model_year_in_both": int(len(set(v["key"]) & keys_c)),
    }


if __name__ == "__main__":
    cmp = load_complaints()
    result = {"complaints": complaint_summary(cmp)}
    result["crss"], veh = crss_summary()
    result["linkage"] = linkage(cmp, veh)
    (TABLE_DIR / "feasibility_summary.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str))
