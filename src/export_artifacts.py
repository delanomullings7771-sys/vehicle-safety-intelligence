"""Copy deployment artifacts into app/api/artifacts and build the model card.

Run after train_crss.py, select_crash_features.py, train_crss_final.py, train_text.py and
build_vehicle_profiles.py.

Crash tool: the deployed crash models are the final models (train_crss_final.py), which take exactly the
features selected in select_crash_features.py. The API accepts a crash record in CRSS format (the crash's rows
from the 12 required tables, in CRSS codes) and builds those features with app/api/crash_features.py.
  * crash_spec.json      the 50 required fields, their official titles, unknown-value sentinels and the
                         training-year code lists the builder needs
  * crash_examples.json  a few real 2024 crashes (the test year) as CRSS records, for the demonstration
Before writing them, this script checks that the API builder reproduces the training features exactly for
3,000 real 2024 crashes, so the deployed models receive exactly what they were trained and evaluated on.
"""
import json
import re
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_crss_features
from project_paths import (CRSS_TEST_YEAR, CRSS_TRAIN_YEARS, EVALUATION_DIR, OUTPUT_DIR, PROCESSED_DIR, PROJECT_ROOT,
                           crss_extracted_dir)

sys.path.insert(0, str(PROJECT_ROOT / "app" / "api"))
import crash_features  # noqa: E402  the API's feature builder, checked below against the training features

ART = PROJECT_ROOT / "app" / "api" / "artifacts"
TARGETS = ("y_injury", "y_serious")
KEYS = ["CASENUM", "VEH_NO", "PER_NO"]
COUNT_TITLES = {"vevent": "Number of vehicle events", "cevent": "Number of crash events", "person": "Number of people"}
TABLE_ORDER = ["accident", "cevent", "vehicle", "drimpair", "distract", "driverrf", "maneuver", "violatn", "vevent",
               "person", "personrf", "pbtype"]


def selected(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    return df[df.selected].iloc[0]


def clean_title(title, field: str) -> str:
    """Official titles without element codes or the 'Imputed' prefix, e.g. 'P6/NM6I. Imputed Sex' -> 'Sex'."""
    if not isinstance(title, str) or not title.strip():
        return field
    t = re.sub(r"^[A-Z0-9/]+[A-Z0-9]*\.\s*", "", title.strip())
    t = re.sub(r"^C\d+\w*\s+", "", t)
    t = re.sub(r"^Imputed\s+", "", t).rstrip(".")
    return t.replace("\u0096", "-").replace("�", "-")


def field_titles(fields: list[str]) -> dict:
    d = pd.read_excel(PROJECT_ROOT / "documentation" / "Data_Dictionary.xlsx", sheet_name="CRSS fields")
    title = d.drop_duplicates(["table", "field"]).set_index(["table", "field"])["title"]
    out = {}
    for k in fields:
        table, field = k.split(".")
        out[k] = (COUNT_TITLES.get(table, f"Number of {table} records") if field == "n_rows"
                  else clean_title(title.get((table, field)), field))
    return out


def pick_examples(test: pd.DataFrame, p1: np.ndarray, p2: np.ndarray, thr: dict) -> list[tuple[int, str]]:
    """A small, varied set of real 2024 crashes for the demonstration, including a miss and a false alarm."""
    t = test.assign(p1=p1, p2=p2)
    picks = []

    def middle(mask, by):
        cand = t[mask & ~t.CASENUM.isin([c for c, _ in picks])].sort_values(by)
        return int(cand.CASENUM.iloc[len(cand) // 2]) if len(cand) else None

    rules = [
        ("No injury recorded; low predicted risk", (t.y_injury == 0) & (t.p1 < 0.15), "p1"),
        ("Injury, not serious; injury flagged", (t.y_injury == 1) & (t.y_serious == 0) & (t.p1 >= thr["y_injury"]) & (t.p2 < thr["y_serious"]), "p1"),
        ("Serious or fatal; high predicted risk", (t.y_serious == 1) & (t.p2 >= 0.8), "p2"),
        ("Serious or fatal; serious flagged", (t.y_serious == 1) & (t.p2 >= thr["y_serious"]) & (t.p2 < 0.6), "p2"),
        ("Serious or fatal, but scored below the threshold (a miss)", (t.y_serious == 1) & (t.p2 < thr["y_serious"] * 0.6), "p2"),
        ("No injury recorded, but flagged (a false alarm)", (t.y_injury == 0) & (t.p1 >= 0.8), "p1"),
    ]
    for label, mask, by in rules:
        c = middle(mask, by)
        if c is not None:
            picks.append((c, label))
    return picks


def raw_tables(fields: list[str]) -> dict[str, pd.DataFrame]:
    """The 2024 rows of each required table: keys, the required fields (CRSS codes) and their NAME labels."""
    by_table = {}
    for k in fields:
        table, field = k.split(".")
        by_table.setdefault(table, []).append(field)
    raw = {}
    for table, flds in by_table.items():
        path = crss_extracted_dir(CRSS_TEST_YEAR) / f"{table}.csv"
        up = {c.upper(): c for c in pd.read_csv(path, nrows=0, encoding="latin-1").columns}
        keep = [up[c] for c in KEYS if c in up]
        for f in flds:
            if f != "n_rows":
                keep += [up[f]] + ([up[f + "NAME"]] if f + "NAME" in up else [])
        t = pd.read_csv(path, usecols=keep, encoding="latin-1", low_memory=False)
        t.columns = [c.upper() for c in t.columns]
        raw[table] = t
    return raw


def crash_record(raw: dict, case: int, fields: list[str]) -> dict:
    """One crash in the API's input format: rows per table, CRSS codes only (no labels, no outcome)."""
    want = {}
    for k in fields:
        table, field = k.split(".")
        want.setdefault(table, [])
        if field != "n_rows":
            want[table].append(field)
    rec = {}
    for table, flds in want.items():
        t = raw[table]
        cols = [c for c in KEYS if c in t.columns] + flds
        rows = t.loc[t.CASENUM == case, cols].astype(object)
        rows = rows.where(rows.notna(), None)
        rec[table] = [{k: (int(v) if isinstance(v, float) and v.is_integer() else v) for k, v in r.items()}
                      for r in rows.to_dict("records")]
    return rec


def crash_display(raw: dict, case: int, fields: list[str], titles: dict) -> list[dict]:
    """The same record with official labels, for people to read."""
    out = []
    for table in TABLE_ORDER:
        flds = [k.split(".")[1] for k in fields if k.startswith(table + ".") and not k.endswith(".n_rows")]
        if table not in raw or not flds:
            continue
        for n, (_, r) in enumerate(raw[table][raw[table].CASENUM == case].iterrows(), start=1):
            if table == "cevent":
                unit = f"Crash event {n}"
            elif "PER_NO" in r.index:
                unit = f"Person {int(r.PER_NO)}" + (f" (vehicle {int(r.VEH_NO)})" if "VEH_NO" in r.index and int(r.VEH_NO) > 0 else "")
            elif "VEH_NO" in r.index:
                unit = f"Vehicle {int(r.VEH_NO)}"
            else:
                unit = "Crash"
            for f in flds:
                value = r[f + "NAME"] if f + "NAME" in r.index else r[f]
                out.append({"table": table, "unit": unit, "field": f, "label": titles[f"{table}.{f}"], "value": str(value)})
    return out


def accident_codes(df: pd.DataFrame, feats: list[str]) -> dict:
    """Training-year code lists for the accident-level coded fields (codes outside them were set to -1)."""
    train = df[df.YEAR.isin(CRSS_TRAIN_YEARS)]
    return {f[5:]: sorted(float(v) for v in train[f].dropna().unique() if v != -1)
            for f in feats if f.startswith("acc__") and f[5:] not in build_crss_features.NUMERIC}


def crash_artifacts(models: dict) -> tuple[dict, dict]:
    sel = json.loads((OUTPUT_DIR / "selection" / "crss_selected_features.json").read_text())
    fields = sel["deployment_fields"]
    titles = field_titles(fields)
    feats = sorted(set(models["y_injury"]["features"]) | set(models["y_serious"]["features"]))
    df = pd.read_parquet(PROCESSED_DIR / "crss_crash_features.parquet")
    used = {k.split(".")[1] for k in fields}
    spec = {"fields": fields, "field_titles": titles,
            "tables": sorted({k.split(".")[0] for k in fields}),
            "keys": {"CASENUM": "crash number; groups rows into one crash when a file holds many crashes",
                     "VEH_NO": "vehicle number; labels vehicle-level rows (optional for scoring)",
                     "PER_NO": "person number; labels person-level rows (optional for scoring)"},
            "numeric_sentinels": {f: v for f, v in build_crss_features.NUMERIC.items() if f in used},
            "accident_codes": accident_codes(df, feats),
            "outcome_fields_ignored": sorted(build_crss_features.LEAKAGE)}

    test = df[(df.YEAR == CRSS_TEST_YEAR) & (df.y_injury >= 0) & (df.y_serious >= 0)].copy()
    raw = raw_tables(fields)

    # Trained = deployed: the API builder must reproduce the training features from raw CRSS rows.
    check = test.sample(3000, random_state=0)
    grouped = {t: dict(tuple(d.groupby("CASENUM"))) for t, d in raw.items()}
    for _, want in check.iterrows():
        case = want.CASENUM
        sub = {t: (g[case] if case in g else raw[t].iloc[:0]) for t, g in grouped.items()}
        got = crash_features.build_features(crash_record(sub, case, fields), feats, spec)
        for f in feats:
            a, b = got[f], float(want[f])
            if not ((np.isnan(a) and np.isnan(b)) or abs(a - b) < 1e-4):
                raise AssertionError(f"crash {case}: {f} built as {a}, training value {b}")
    print(f"Feature builder reproduces the training features for {len(check):,} 2024 crashes")

    prob = {t: models[t]["pipeline"].predict_proba(test[models[t]["features"]])[:, 1] for t in TARGETS}
    picks = pick_examples(test, prob["y_injury"], prob["y_serious"], {t: models[t]["threshold"] for t in TARGETS})
    acc = pd.read_csv(crss_extracted_dir(CRSS_TEST_YEAR) / "accident.csv", usecols=["CASENUM", "MAX_SEVNAME"], encoding="latin-1")
    examples = []
    for i, (case, label) in enumerate(picks):
        row = test[test.CASENUM == case].iloc[0]
        examples.append({"id": f"crash-{i + 1}", "casenum": str(case), "year": int(CRSS_TEST_YEAR), "label": label,
                         "recorded_outcome": acc.loc[acc.CASENUM == case, "MAX_SEVNAME"].iloc[0],
                         "y_injury": int(row.y_injury), "y_serious": int(row.y_serious),
                         "record": crash_record(raw, case, fields),
                         "display": crash_display(raw, case, fields, titles),
                         "training_features": {f: (None if pd.isna(row[f]) else float(row[f])) for f in feats}})
    return spec, {"examples": examples}


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    for task in ("U1", "U2"):
        bundle = joblib.load(PROJECT_ROOT / "models" / f"text_{task}.joblib")
        # stop_words_ only records terms pruned during fitting; it is not used for transform().
        bundle["vectorizer"].stop_words_ = None
        joblib.dump(bundle, ART / f"text_{task}.joblib", compress=3)

    models = {t: joblib.load(PROJECT_ROOT / "models" / f"crss_{t}_final.joblib") for t in TARGETS}
    for t in TARGETS:
        joblib.dump(models[t], ART / f"crash_{t}.joblib", compress=3)
    shutil.copy(Path(__file__).parent / "crss_transformers.py", ART.parent / "crss_transformers.py")  # loads the pipelines
    spec, examples = crash_artifacts(models)
    (ART / "crash_spec.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
    (ART / "crash_examples.json").write_text(json.dumps(examples, indent=1), encoding="utf-8")

    prof = pd.read_parquet(PROCESSED_DIR / "vehicle_profiles.parquet")
    prof.to_parquet(ART / "vehicle_profiles.parquet", index=False)

    final = {t: pd.read_csv(EVALUATION_DIR / "final" / f"crss_{t}_final_results.csv") for t in TARGETS}
    fin = {t: f[f.model != "Dummy (prior)"].iloc[0] for t, f in final.items()}
    base = {t: f[f.model == "Dummy (prior)"].iloc[0].test_pr_auc for t, f in final.items()}
    u1, u2 = (selected(EVALUATION_DIR / f"text_{t}_model_comparison.csv") for t in ("U1", "U2"))
    card = {"models": [
        {"task": "S1 Injury crash (CRSS)", "model": "Gradient Boosting", "headline_metric": "PR-AUC, 2024 test",
         "headline_value": f"{fin['y_injury'].test_pr_auc:.3f}",
         "detail": f"baseline {base['y_injury']:.3f}; ROC-AUC {fin['y_injury'].test_roc_auc:.3f}; "
                   f"{int(fin['y_injury'].n_features)} features from {int(fin['y_injury'].n_fields)} CRSS fields"},
        {"task": "S2 Serious or fatal crash (CRSS)", "model": "Gradient Boosting", "headline_metric": "PR-AUC, 2024 test",
         "headline_value": f"{fin['y_serious'].test_pr_auc:.3f}",
         "detail": f"baseline {base['y_serious']:.3f}; ROC-AUC {fin['y_serious'].test_roc_auc:.3f}; "
                   f"{int(fin['y_serious'].n_features)} features from {int(fin['y_serious'].n_fields)} CRSS fields"},
        {"task": "U1 Component routing (complaints)", "model": u1.model, "headline_metric": "Top-1 accuracy, 2024-26 test",
         "headline_value": f"{u1.test_top1_accuracy:.1%}",
         "detail": f"micro-F1 {u1.test_micro_f1:.3f}, macro-F1 {u1.test_macro_f1:.3f} across 19 groups"},
        {"task": "U2 Serious-incident flag (complaints)", "model": u2.model, "headline_metric": "ROC-AUC, 2024-26 test",
         "headline_value": f"{u2.test_roc_auc:.3f}",
         "detail": f"PR-AUC {u2.test_pr_auc:.3f}, recall {u2.test_recall:.2f}, precision {u2.test_precision:.2f}"},
    ]}
    (ART / "model_card.json").write_text(json.dumps(card, indent=1), encoding="utf-8")
    for f in sorted(ART.iterdir()):
        print(f"{f.name:32s} {f.stat().st_size / 1e6:8.1f} MB")
    print(json.dumps(card, indent=1))


if __name__ == "__main__":
    main()
