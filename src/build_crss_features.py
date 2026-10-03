"""Build one-row-per-crash CRSS analytical table from all 28 tables, 2020-2024.

The 26 crash, vehicle and person tables describe what happened; the two vPIC tables hold the
manufacturer specification decoded from each vehicle's VIN (fitted safety systems, weight, power).

Every documented field is considered. Fields are removed only when they are identifiers,
sampling-design variables or encodings of the injury outcome (see LEAKAGE / DESIGN below).
Child tables (vehicle, person and multi-row risk-factor tables) are aggregated to crash
level as counts of units carrying each frequent code; code vocabularies are learned from
training years only (2020-2022) so validation/test years cannot influence feature design.

Outputs
  data/processed/crss_crash_features.parquet
  outputs/tables/crss_feature_register.csv   (every feature, its source and group)
  outputs/tables/crss_excluded_fields.csv    (every excluded field and the reason)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from project_paths import CRSS_TRAIN_YEARS, CRSS_YEARS, PROCESSED_DIR, TABLE_DIR, crss_extracted_dir

TOP_K = 12          # most frequent codes per field (learned on training years)
MIN_SHARE = 0.002   # ignore codes rarer than this in training data

KEYS = {"CASENUM", "VEH_NO", "PER_NO", "EVENTNUM", "VEVENTNUM", "VNUMBER1", "VNUMBER2", "VNumber2", "TRAILER_NO"}
DESIGN = {"PSU", "PSU_VAR", "PSUSTRAT", "STRATUM", "PJ", "WEIGHT", "YEAR"}
# Fields that encode the outcome being predicted (target leakage).
LEAKAGE = {
    "MAX_SEV": "defines the target", "MAXSEV_IM": "imputed target",
    "NUM_INJ": "injury count", "NO_INJ_IM": "imputed injury count",
    "MAX_VSEV": "vehicle max injury severity", "MXVSEV_IM": "imputed vehicle max severity",
    "NUM_INJV": "vehicle injury count", "NUMINJ_IM": "imputed vehicle injury count",
    "INJ_SEV": "person injury severity", "INJSEV_IM": "imputed person injury severity",
    "HOSPITAL": "transport to hospital follows injury",
    "PVEH_SEV": "parked-vehicle injury severity",
}
# Identifiers / free text / extreme-cardinality codes that carry no generalisable meaning.
IDENTIFIERS = {
    "VIN": "vehicle identifier", "TRLR1VIN": "identifier", "TRLR2VIN": "identifier",
    "TRLR3VIN": "identifier", "MCARR_ID": "carrier identifier", "MCARR_I1": "carrier identifier",
    "MCARR_I2": "carrier identifier", "DR_ZIP": "driver ZIP code", "MAK_MOD": "make-model code (kept via MAKE)",
    "MODEL": "model code (kept via MAKE)", "VPICMAKE": "duplicate of MAKE", "VPICMODEL": "model identifier",
    "HAZ_ID": "hazmat identifier", "HAZ_CNO": "hazmat class number", "MINUTE": "clock minute",
    "MINUTE_IM": "clock minute", "PVIN": "identifier", "PTRLR1VIN": "identifier", "PTRLR2VIN": "identifier",
    "PTRLR3VIN": "identifier", "PMCARR_ID": "identifier", "PMCARR_I1": "identifier", "PMCARR_I2": "identifier",
    "PMAK_MOD": "identifier", "PMODEL": "identifier", "PVPICMAKE": "identifier", "PVPICMODEL": "identifier",
    "PHAZ_ID": "identifier", "PHAZ_CNO": "identifier", "PMINUTE": "clock minute",
    # vPIC VIN-decode tables
    "VEHICLEDESCRIPTOR": "partial VIN pattern", "VINDECODEDON": "decode timestamp",
    "MAKEID": "make identifier (make kept via MAKE)", "MODELID": "model identifier",
    "MANUFACTURERFULLNAMEID": "manufacturer identifier (make kept via MAKE)",
    "MODELYEAR": "duplicate of MOD_YEAR", "DISPLACEMENTCI": "duplicate of DISPLACEMENTL (other unit)",
    "DISPLACEMENTCC": "duplicate of DISPLACEMENTL (other unit)",
}
VPIC_TABLES = {"vpicdecode", "vpictrailerdecode"}
# Recorded after the crash by investigation, kept but tagged for sensitivity analysis.
POST_CRASH = {"DEFORMED", "TOWED", "DAMAGE", "ALC_RES", "ATST_TYP", "ALC_STATUS", "DRUG_RES",
              "PTOWED", "PIMPACT1"}
# Numeric fields with documented unknown/not-applicable sentinels (value >= sentinel -> missing).
NUMERIC = {"NUMOCCS": 97, "MOD_YEAR": 9998, "MDLYR_IM": 9998, "TRAV_SP": 997, "VSPD_LIM": 98,
           "AGE": 998, "AGE_IM": 998, "PEDS": 10**6, "PERNOTMVIT": 10**6, "VE_TOTAL": 10**6,
           "VE_FORMS": 10**6, "PVH_INVL": 10**6, "PERMVIT": 10**6, "PNUMOCCS": 97, "PMODYEAR": 9998,
           "PBAGE": 998}
# Accident fields repeated on child tables (already represented at crash level).
REPEATED = {"MONTH", "HOUR", "MINUTE", "HARM_EV", "MAN_COLL", "SCH_BUS", "VE_FORMS", "REGION",
            "URBANICITY", "DAY_WEEK"}

TABLE_LEVEL = {  # table -> grain
    "accident": "crash", "crashrf": "crash", "weather": "crash", "cevent": "crash",
    "vehicle": "vehicle", "damage": "vehicle", "distract": "vehicle", "drimpair": "vehicle",
    "driverrf": "vehicle", "factor": "vehicle", "maneuver": "vehicle", "vehiclesf": "vehicle",
    "violatn": "vehicle", "vision": "vehicle", "vevent": "vehicle", "vsoe": "vehicle",
    "parkwork": "vehicle", "pvehiclesf": "vehicle",
    "person": "person", "nmcrash": "person", "nmdistract": "person", "nmimpair": "person",
    "nmprior": "person", "personrf": "person", "safetyeq": "person", "pbtype": "person",
    "vpicdecode": "vehicle", "vpictrailerdecode": "vehicle",
}


def read(year: int, table: str) -> pd.DataFrame:
    df = pd.read_csv(crss_extracted_dir(year) / f"{table}.csv", encoding="latin-1", low_memory=False)
    df = df[[c for c in df.columns if not c.endswith("NAME")]]
    df.columns = [c.upper() for c in df.columns]
    df["YEAR"] = year
    return df


def field_lists(table: str, df: pd.DataFrame) -> tuple[list[str], list[str]]:
    cols = [c for c in df.columns if c not in KEYS | DESIGN | set(LEAKAGE) | set(IDENTIFIERS)]
    if table != "accident":
        cols = [c for c in cols if c not in REPEATED]
    # Where NHTSA supplies an imputed version, use it and drop the raw field.
    imputed_raw = {"HOUR_IM": "HOUR", "WKDY_IM": "DAY_WEEK", "EVENT1_IM": "HARM_EV", "MANCOL_IM": "MAN_COLL",
                   "RELJCT1_IM": "RELJCT1", "RELJCT2_IM": "RELJCT2", "LGTCON_IM": "LGT_COND",
                   "WEATHR_IM": "WEATHER", "ALCHL_IM": "ALCOHOL", "MDLYR_IM": "MOD_YEAR",
                   "IMPACT1_IM": "IMPACT1", "VEVENT_IM": "M_HARM", "V_ALCH_IM": "VEH_ALCH",
                   "PCRASH1_IM": "P_CRASH1", "AGE_IM": "AGE", "SEX_IM": "SEX", "SEAT_IM": "SEAT_POS",
                   "EJECT_IM": "EJECTION", "PERALCH_IM": "DRINKING"}
    if table in {"accident", "vehicle", "person"}:
        for imp, raw in imputed_raw.items():
            if imp in cols and raw in cols:
                cols.remove(raw)
    if table in VPIC_TABLES:
        # Text columns are labels of the coded *ID columns or free notes; codes and measurements are used.
        cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        categorical = [c for c in cols if c.endswith("ID")]
        return [c for c in cols if c not in categorical], categorical
    numeric = [c for c in cols if c in NUMERIC]
    categorical = [c for c in cols if c not in NUMERIC]
    return numeric, categorical


def clean_numeric(s: pd.Series, field: str) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return s.where(s < NUMERIC.get(field, np.inf))


def learn_vocab(train: pd.DataFrame, categorical: list[str]) -> dict[str, list]:
    vocab = {}
    for c in categorical:
        share = train[c].value_counts(normalize=True)
        vocab[c] = share[share >= MIN_SHARE].index[:TOP_K].tolist()
    return vocab


def aggregate(table: str, df: pd.DataFrame, numeric, vocab) -> pd.DataFrame:
    key = ["YEAR", "CASENUM"]
    parts = []
    if table == "accident":
        out = df[key].copy()
        for c in numeric:
            out[f"acc__{c}"] = clean_numeric(df[c], c).astype("float32")
        for c, codes in vocab.items():
            vals = df[c].where(df[c].isin(codes), other=-1)
            out[f"acc__{c}"] = vals.astype("float32")  # categorical code, one-hot later
        return out.set_index(key)
    prefix = table
    grouped = df.groupby(key)
    parts.append(grouped.size().rename(f"{prefix}__n_rows").astype("float32"))
    for c in numeric:
        v = clean_numeric(df[c], c)
        g = v.groupby([df["YEAR"], df["CASENUM"]])
        parts += [g.min().rename(f"{prefix}__{c}__min").astype("float32"),
                  g.max().rename(f"{prefix}__{c}__max").astype("float32")]
    for c, codes in vocab.items():
        for code in codes:
            ind = (df[c] == code).astype("uint8")
            parts.append(ind.groupby([df["YEAR"], df["CASENUM"]]).sum()
                         .rename(f"{prefix}__{c}={code}").astype("float32"))
    return pd.concat(parts, axis=1)


def main() -> None:
    data = {t: {y: read(y, t) for y in CRSS_YEARS} for t in TABLE_LEVEL}
    register, excluded = [], []
    frames = []
    for table, by_year in data.items():
        union = pd.concat(by_year.values(), ignore_index=True)
        numeric, categorical = field_lists(table, union)
        for c in union.columns:
            reason = (LEAKAGE.get(c) or IDENTIFIERS.get(c)
                      or ("sampling design / split variable" if c in DESIGN else None)
                      or ("key" if c in KEYS else None)
                      or ("text label of a coded *ID field, or free-text note"
                          if table in VPIC_TABLES and not pd.api.types.is_numeric_dtype(union[c]) else None))
            if reason:
                excluded.append({"table": table, "field": c, "reason": reason})
        train = union[union["YEAR"].isin(CRSS_TRAIN_YEARS)]
        vocab = learn_vocab(train, categorical)
        agg = aggregate(table, union, numeric, vocab)
        for col in agg.columns:
            field = col.split("__")[1].split("=")[0]
            register.append({"feature": col, "table": table, "grain": TABLE_LEVEL[table], "field": field,
                             "group": "post_crash" if field in POST_CRASH or table == "damage" else "crash_record",
                             "kind": "categorical_code" if table == "accident" and field in vocab else
                             ("count" if "=" in col or col.endswith("n_rows") else "numeric")})
        frames.append(agg)
        print(f"{table:10s} rows={len(union):>9,} numeric={len(numeric):>3} categorical={len(categorical):>3} features={agg.shape[1]:>4}")

    acc = data["accident"]
    acc_all = pd.concat(acc.values(), ignore_index=True)
    base = acc_all[["YEAR", "CASENUM", "WEIGHT", "MAX_SEV"]].set_index(["YEAR", "CASENUM"])
    X = pd.concat([base] + frames, axis=1)
    # A crash with no rows in a child table has zero units with each code; min/max stay missing.
    child_counts = [c for c in X.columns if "__" in c and not c.startswith("acc__")
                    and not c.endswith(("__min", "__max"))]
    X[child_counts] = X[child_counts].fillna(0)
    sev = X["MAX_SEV"]
    X["y_injury"] = np.select([sev == 0, sev.isin([1, 2, 3, 4, 5])], [0, 1], default=-1).astype("int8")
    X["y_serious"] = np.select([sev.isin([0, 1, 2]), sev.isin([3, 4])], [0, 1], default=-1).astype("int8")
    X = X.drop(columns="MAX_SEV").reset_index()
    assert X.duplicated(["YEAR", "CASENUM"]).sum() == 0, "crash duplicated during aggregation"
    assert len(X) == len(acc_all), "row conservation failed"
    X.to_parquet(PROCESSED_DIR / "crss_crash_features.parquet", index=False)
    pd.DataFrame(register).to_csv(TABLE_DIR / "crss_feature_register.csv", index=False)
    pd.DataFrame(excluded).drop_duplicates(["table", "field"]).to_csv(TABLE_DIR / "crss_excluded_fields.csv", index=False)
    print("crashes:", len(X), "features:", len(register))
    print(X.groupby("YEAR")[["y_injury", "y_serious"]].agg(lambda s: f"{(s == 1).mean():.3f} (excl {(s == -1).sum()})"))


if __name__ == "__main__":
    main()
