"""Compact crash-severity models for the web application.

The research models use ~1,000 crash-level features; an analyst cannot enter those by hand.
This script derives ~30 human-enterable inputs (police-report level), retrains Logistic Regression
and Gradient Boosting on the same chronological split, and records how much accuracy the
compact form costs relative to the full research model.
Outputs: app/api/artifacts/crash_<target>.joblib, app/api/artifacts/crash_form.json,
         outputs/evaluation/crash_deploy_comparison.csv
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from project_paths import (CRSS_TEST_YEAR, CRSS_TRAIN_YEARS, CRSS_VALIDATION_YEAR, CRSS_YEARS, EVALUATION_DIR,
                           PROCESSED_DIR, PROJECT_ROOT, crss_extracted_dir)

ART = PROJECT_ROOT / "app" / "api" / "artifacts"
RNG = 42

CATEGORICAL = {  # form field -> (feature column, label, help)
    "first_harmful_event": ("acc__EVENT1_IM", "First harmful event", "what the vehicle first struck", "EVENT1_IM", "HARM_EV"),
    "manner_of_collision": ("acc__MANCOL_IM", "Manner of collision", "between motor vehicles", "MANCOL_IM", "MAN_COLL"),
    "relation_to_road": ("acc__REL_ROAD", "Location relative to road", None, "REL_ROAD", "REL_ROAD"),
    "intersection_type": ("acc__TYP_INT", "Intersection type", None, "TYP_INT", "TYP_INT"),
    "junction": ("acc__RELJCT2_IM", "Relation to junction", None, "RELJCT2_IM", "RELJCT2"),
    "light": ("acc__LGTCON_IM", "Light condition", None, "LGTCON_IM", "LGT_COND"),
    "weather": ("acc__WEATHR_IM", "Weather", None, "WEATHR_IM", "WEATHER"),
    "alcohol": ("acc__ALCHL_IM", "Alcohol involved", "police-reported", "ALCHL_IM", "ALCOHOL"),
    "work_zone": ("acc__WRK_ZONE", "Work zone", None, "WRK_ZONE", "WRK_ZONE"),
    "interstate": ("acc__INT_HWY", "Interstate highway", None, "INT_HWY", "INT_HWY"),
    "urbanicity": ("acc__URBANICITY", "Area", None, "URBANICITY", "URBANICITY"),
    "region": ("acc__REGION", "US region", None, "REGION", "REGION"),
    "weekday": ("acc__WKDY_IM", "Day of week", None, "WKDY_IM", "DAY_WEEK"),
}
NUMERIC = {  # form field -> (builder, label, help, min, max)
    "hour": (lambda d: d["acc__HOUR_IM"].where(d["acc__HOUR_IM"] < 24), "Hour of day", "0-23", 0, 23),
    "vehicles": (lambda d: d["acc__VE_TOTAL"], "Vehicles involved", None, 1, 20),
    "occupants": (lambda d: d["acc__PERMVIT"], "Vehicle occupants", "all vehicles", 0, 60),
    "pedestrians_cyclists": (lambda d: d["acc__PEDS"], "Pedestrians / cyclists involved", None, 0, 20),
    "max_travel_speed": (lambda d: d["vehicle__TRAV_SP__max"], "Highest travel speed (mph)", "of any vehicle", 0, 150),
    "speed_limit": (lambda d: d["vehicle__VSPD_LIM__max"], "Speed limit (mph)", None, 5, 85),
    "oldest_model_year": (lambda d: d["vehicle__MDLYR_IM__min"], "Oldest vehicle model year", None, 1950, 2026),
    "youngest_person_age": (lambda d: d["person__AGE_IM__min"], "Youngest person's age", None, 0, 110),
    "oldest_person_age": (lambda d: d["person__AGE_IM__max"], "Oldest person's age", None, 0, 110),
    "rollovers": (lambda d: d.filter(regex=r"^vehicle__ROLLOVER=(1|2|3)$").sum(1), "Vehicles that rolled over", None, 0, 10),
    "unrestrained_occupants": (lambda d: d["person__REST_USE=20"], "Occupants not using a restraint", None, 0, 30),
    "ejected": (lambda d: d["person__EJECT_IM=1"], "People ejected", None, 0, 20),
    "motorcycles": (lambda d: d["vehicle__BODY_TYP=80"], "Motorcycles involved", None, 0, 10),
    "truck_tractors": (lambda d: d["vehicle__BODY_TYP=66"], "Truck-tractors involved", None, 0, 10),
    "hit_and_run": (lambda d: d["vehicle__HIT_RUN=1"], "Hit-and-run vehicles", None, 0, 5),
    "speeding_vehicles": (lambda d: d.filter(regex=r"^vehicle__SPEEDREL=(3|4|5)$").sum(1), "Vehicles speeding", "police-reported", 0, 10),
    "vehicle_fire": (lambda d: d["vehicle__FIRE_EXP=1"], "Vehicles with fire", None, 0, 5),
}


def build(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for f, (col, *_rest) in CATEGORICAL.items():
        X[f] = df[col].where(df[col] >= 0)  # -1 = code outside training vocabulary -> unknown
    for f, (fn, *_rest) in NUMERIC.items():
        X[f] = fn(df).astype("float32")
    return X


def code_labels(field_im: str, field_raw: str) -> dict:
    """Official labels for codes from the accident files, latest year first (imputed fields share raw labels)."""
    labels = {}
    for year in sorted(CRSS_YEARS, reverse=True):
        a = pd.read_csv(crss_extracted_dir(year) / "accident.csv", encoding="latin-1", low_memory=False,
                        keep_default_na=False)  # "None" is a real CRSS label, not missing
        a.columns = [c.upper() for c in a.columns]
        for src in (field_im, field_raw):
            if src + "NAME" not in a.columns:
                continue
            pairs = a[[src, src + "NAME"]].dropna().drop_duplicates()
            for code, name in zip(pairs[src].astype(int), pairs[src + "NAME"].astype(str)):
                labels.setdefault(code, name)
    return labels


def best_threshold(y, p) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    return float(thr[np.nanargmax(f1[:-1])])


def main() -> None:
    df = pd.read_parquet(PROCESSED_DIR / "crss_crash_features.parquet")
    X_all = build(df)
    cats, nums = list(CATEGORICAL), list(NUMERIC)
    ART.mkdir(parents=True, exist_ok=True)
    rows = []

    # Form specification: options are the codes the model saw in training, labelled officially.
    train_mask = df.YEAR.isin(CRSS_TRAIN_YEARS)
    fields = []
    for f, (col, label, help_, im, raw) in CATEGORICAL.items():
        labels = code_labels(im, raw)
        codes = sorted(X_all.loc[train_mask, f].dropna().astype(int).unique())
        fields.append({"name": f, "type": "select", "label": label, "help": help_,
                       "options": [{"value": int(c), "label": labels.get(int(c), f"Code {c}")} for c in codes
                                   if "unknown" not in labels.get(int(c), "").lower()
                                   and "not reported" not in labels.get(int(c), "").lower()]})
    for f, (_fn, label, help_, lo, hi) in NUMERIC.items():
        fields.append({"name": f, "type": "number", "label": label, "help": help_, "min": lo, "max": hi})
    (ART / "crash_form.json").write_text(json.dumps({"fields": fields}, indent=1), encoding="utf-8")

    for target in ["y_injury", "y_serious"]:
        d = df[target] >= 0
        tr, va, te = (d & df.YEAR.isin(CRSS_TRAIN_YEARS), d & (df.YEAR == CRSS_VALIDATION_YEAR), d & (df.YEAR == CRSS_TEST_YEAR))
        y = df[target]
        models = {
            "Logistic Regression": Pipeline([
                ("prep", ColumnTransformer([
                    ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value=-9)),
                                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), cats),
                    ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                                      ("sc", StandardScaler())]), nums)])),
                ("model", LogisticRegression(C=0.1, max_iter=3000))]),
            "Gradient Boosting": Pipeline([
                ("prep", ColumnTransformer([
                    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                                           encoded_missing_value=np.nan), cats),
                    ("num", "passthrough", nums)])),
                ("model", HistGradientBoostingClassifier(learning_rate=0.08, max_leaf_nodes=31, max_iter=800,
                                                         categorical_features=list(range(len(cats))),
                                                         l2_regularization=1.0, early_stopping=True,
                                                         validation_fraction=0.1, n_iter_no_change=30, random_state=RNG))]),
        }
        best = None
        for name, pipe in models.items():
            pipe.fit(X_all[tr], y[tr])
            pv, pt, ptr = (pipe.predict_proba(X_all[m])[:, 1] for m in (va, te, tr))
            thr = best_threshold(y[va], pv)
            row = {"target": target, "model": f"Compact {name}", "inputs": len(cats) + len(nums),
                   "train_roc_auc": roc_auc_score(y[tr], ptr), "validation_roc_auc": roc_auc_score(y[va], pv),
                   "validation_pr_auc": average_precision_score(y[va], pv), "test_roc_auc": roc_auc_score(y[te], pt),
                   "test_pr_auc": average_precision_score(y[te], pt), "test_f1": f1_score(y[te], pt >= thr), "threshold": thr}
            rows.append(row)
            print(row, flush=True)
            if best is None or row["validation_pr_auc"] > best[0]["validation_pr_auc"]:
                best = (row, pipe)
        row, pipe = best
        joblib.dump({"pipeline": pipe, "features": cats + nums, "threshold": row["threshold"], "target": target,
                     "model": row["model"], "test_metrics": {k: v for k, v in row.items() if k.startswith("test_")},
                     "labels": {f["name"]: f["label"] for f in fields}},
                    ART / f"crash_{target}.joblib", compress=3)
    res = pd.DataFrame(rows)
    for target in ["y_injury", "y_serious"]:
        full = EVALUATION_DIR / f"crss_{target}_model_comparison.csv"
        if full.exists():
            f = pd.read_csv(full)
            sel = f[f.selected].iloc[0]
            res.loc[res.target == target, "full_model_test_roc_auc"] = sel.test_roc_auc
            res.loc[res.target == target, "full_model"] = f"{sel.model} ({sel.n_input_fields} fields)"
    res.to_csv(EVALUATION_DIR / "crash_deploy_comparison.csv", index=False)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
