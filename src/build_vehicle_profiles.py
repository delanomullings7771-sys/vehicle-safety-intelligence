"""Integration layer: per-vehicle safety profiles combining both datasets.

Join key: normalised make + model + model year (aggregate level; no record-level linkage).
Complaint side (2020-2024 received, matching the CRSS window): complaint volume, serious-incident
share, component mix. CRSS side: survey-weighted crash involvement (exposure proxy), injury and
serious/fatal crash rates, and police-recorded vehicle-defect involvement.
Output: data/processed/vehicle_profiles.parquet (consumed by the web API).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from project_paths import CRSS_YEARS, PROCESSED_DIR, TABLE_DIR, crss_extracted_dir

WINDOW = (2020, 2024)
MIN_CRSS_SAMPLE = 20   # below this, CRSS-based rates are suppressed as unreliable


def norm(s: pd.Series) -> pd.Series:
    return s.astype("string").str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)


def crss_side() -> pd.DataFrame:
    rows = []
    for y in CRSS_YEARS:
        d = crss_extracted_dir(y)
        v = pd.read_csv(d / "vehicle.csv", encoding="latin-1", low_memory=False,
                        usecols=["CASENUM", "VEH_NO", "MOD_YEAR", "VPICMAKENAME", "VPICMODELNAME", "WEIGHT"])
        a = pd.read_csv(d / "accident.csv", encoding="latin-1", usecols=["CASENUM", "MAX_SEV"])
        f = pd.read_csv(d / "factor.csv", encoding="latin-1", usecols=["CASENUM", "VEH_NO", "VEHICLECC"])
        f = f[~f.VEHICLECC.isin([0, 99])].drop_duplicates(["CASENUM", "VEH_NO"]).assign(defect=1)
        v = v.merge(a, on="CASENUM").merge(f[["CASENUM", "VEH_NO", "defect"]], on=["CASENUM", "VEH_NO"], how="left")
        rows.append(v)
    v = pd.concat(rows)
    v = v[v.MOD_YEAR.between(1990, 2026) & v.MAX_SEV.isin([0, 1, 2, 3, 4])]
    v["injury"] = v.MAX_SEV.between(1, 4).astype(float)
    v["serious"] = v.MAX_SEV.isin([3, 4]).astype(float)
    v["defect"] = v.defect.fillna(0)
    v["mk"], v["md"] = norm(v.VPICMAKENAME), norm(v.VPICMODELNAME)
    w = v.WEIGHT

    def agg(g):
        ww = g.WEIGHT
        return pd.Series({"crss_sample_vehicles": len(g), "est_crash_involved": ww.sum(),
                          "crash_injury_rate": np.average(g.injury, weights=ww),
                          "crash_serious_rate": np.average(g.serious, weights=ww),
                          "crash_defect_rate": np.average(g.defect, weights=ww),
                          "crss_make_name": g.VPICMAKENAME.mode().iat[0], "crss_model_name": g.VPICMODELNAME.mode().iat[0]})
    return v.groupby(["mk", "md", "MOD_YEAR"]).apply(agg, include_groups=False)


def complaint_side() -> pd.DataFrame:
    c = pd.read_parquet(PROCESSED_DIR / "complaints_model.parquet")
    groups = [x for x in c.columns if x.startswith("c__")]
    c = c[c.year.between(*WINDOW)].copy()
    c["mk"], c["md"] = norm(c["make"]), norm(c["model"])
    c["MOD_YEAR"] = pd.to_numeric(c.model_year, errors="coerce")
    c = c.dropna(subset=["MOD_YEAR"])
    out = c.groupby(["mk", "md", "MOD_YEAR"]).agg(
        make=("make", lambda s: s.mode().iat[0]), model=("model", lambda s: s.mode().iat[0]),
        complaints=("ODINO", "size"), serious_complaints=("y_serious", "sum"),
        crash_complaints=("crash", "sum"), fire_complaints=("fire", "sum"), injury_complaints=("injured", "sum"),
        **{g: (g, "sum") for g in groups})
    out["serious_share"] = out.serious_complaints / out.complaints
    return out


def main() -> None:
    crss = crss_side()
    comp = complaint_side()
    prof = comp.join(crss, how="left")
    prof.index = prof.index.set_names(["mk", "md", "model_year"])
    ok = prof.crss_sample_vehicles >= MIN_CRSS_SAMPLE
    for col in ["crash_injury_rate", "crash_serious_rate", "crash_defect_rate"]:
        prof.loc[~ok, col] = np.nan
    prof["complaints_per_10k_crash_involved"] = np.where(ok, prof.complaints / prof.est_crash_involved * 1e4, np.nan)
    # Percentile of exposure-adjusted complaint rate among vehicles with reliable exposure.
    prof["complaint_rate_percentile"] = prof.complaints_per_10k_crash_involved.rank(pct=True)
    prof = prof.reset_index()
    prof.to_parquet(PROCESSED_DIR / "vehicle_profiles.parquet", index=False)
    summary = {"vehicle_profiles": len(prof), "with_reliable_crss_exposure": int(ok.sum()),
               "complaints_covered": int(prof.complaints.sum()),
               "complaints_in_profiles_with_exposure": int(prof.loc[ok.values, "complaints"].sum())}
    pd.Series(summary).to_csv(TABLE_DIR / "vehicle_profiles_summary.csv")
    print(summary)
    print(prof[ok.values].sort_values("complaints", ascending=False)
          .head(15)[["make", "model", "model_year", "complaints", "serious_share", "est_crash_involved",
                     "complaints_per_10k_crash_involved", "crash_serious_rate"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
