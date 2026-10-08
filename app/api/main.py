"""Vehicle Safety Intelligence API.

Serves the trained models as three human-review workflows:
  /api/complaint/analyze   U1 component routing + U2 serious-incident flag for a narrative
  /api/crash/assess        S1 injury + S2 serious/fatal likelihood for a crash record in CRSS format
  /api/crash/schema        the 50 CRSS fields (12 tables) a crash record must contain
  /api/crash/examples      real 2024 crashes (the test year) in that format, for demonstration
  /api/vehicles/...        integrated make/model/year profile (complaints + CRSS exposure/severity)
Artifacts are produced by src/export_artifacts.py into app/api/artifacts.
"""
import json
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import crash_features  # builds the final crash models' features from a CRSS record, as in training

ART = Path(os.environ.get("ARTIFACT_DIR", Path(__file__).parent / "artifacts"))
DISCLAIMER = ("Decision support for human review only. Outputs do not establish causation, liability "
              "or the existence of a defect.")

app = FastAPI(title="Vehicle Safety Intelligence API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
                   allow_methods=["GET", "POST"], allow_headers=["*"])


@lru_cache
def text_model(task: str) -> dict:
    return joblib.load(ART / f"text_{task}.joblib")


@lru_cache
def crash_model(target: str) -> dict:
    return joblib.load(ART / f"crash_{target}.joblib")


@lru_cache
def crash_spec() -> dict:
    return json.loads((ART / "crash_spec.json").read_text(encoding="utf-8"))


@lru_cache
def crash_examples() -> dict:
    return {e["id"]: e for e in json.loads((ART / "crash_examples.json").read_text(encoding="utf-8"))["examples"]}


@lru_cache
def profiles() -> pd.DataFrame:
    return pd.read_parquet(ART / "vehicle_profiles.parquet")


@lru_cache
def model_card() -> dict:
    return json.loads((ART / "model_card.json").read_text(encoding="utf-8"))


def _scores(model, X) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X))
    return np.asarray(model.predict_proba(X))[:, 1]


def _term_contributions(bundle: dict, x, estimator, top: int = 8) -> list[dict]:
    """Terms in this narrative with the largest positive weight x tf-idf contribution."""
    coef = estimator.coef_.ravel() if hasattr(estimator, "coef_") else (
        estimator.feature_log_prob_[1] - estimator.feature_log_prob_[0])
    row = x.tocsr()
    contrib = row.data * coef[row.indices]
    order = np.argsort(-contrib)[:top]
    terms = bundle["vectorizer"].get_feature_names_out()
    return [{"term": str(terms[row.indices[i]]), "weight": round(float(contrib[i]), 4)} for i in order if contrib[i] > 0]


class Narrative(BaseModel):
    text: str = Field(..., min_length=20, max_length=20000, description="Complaint narrative")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/meta")
def meta():
    return {**model_card(), "disclaimer": DISCLAIMER}


@app.post("/api/complaint/analyze")
def analyze_complaint(body: Narrative):
    u1, u2 = text_model("U1"), text_model("U2")
    x1 = u1["vectorizer"].transform([body.text])
    s1 = _scores(u1["model"], x1).ravel()
    margin = s1 - u1["threshold"]
    order = np.argsort(-margin)
    components = []
    for rank, j in enumerate(order[:5]):
        est = u1["model"].estimators_[j]
        components.append({"component": u1["labels"][j], "score": round(float(s1[j]), 4),
                           "threshold": round(float(u1["threshold"][j]), 4),
                           "recommended": bool(margin[j] >= 0 or rank == 0),
                           "evidence": _term_contributions(u1, x1, est)})
    x2 = u2["vectorizer"].transform([body.text])
    s2 = float(_scores(u2["model"], x2).ravel()[0])
    return {
        "components": components,
        "serious_incident": {"score": round(s2, 4), "threshold": round(float(u2["threshold"]), 4),
                             "flagged": bool(s2 >= u2["threshold"]),
                             "evidence": _term_contributions(u2, x2, u2["model"])},
        "models": {"routing": u1["model_name"], "serious_flag": u2["model_name"]},
        "disclaimer": DISCLAIMER,
    }


class CrashRecord(BaseModel):
    record: dict[str, list[dict[str, Any]]] = Field(
        ..., description="One crash: rows per CRSS table, in CRSS codes, e.g. {'accident': [{...}], 'person': [...]}")


CRASH_NOTE = "Probability is relative to the CRSS sample, which over-represents injury crashes."


@app.get("/api/crash/schema")
def crash_schema():
    spec = crash_spec()
    tables = {}
    for key in spec["fields"]:
        table, field = key.split(".")
        tables.setdefault(table, []).append({"field": field, "title": spec["field_titles"][key]})
    return {"format": "One crash per request: {'record': {table: [rows]}}, CRSS codes, field names as in the CRSS files. "
                      "A table given as an empty list means the crash has no such rows; an absent table is treated as missing.",
            "tables": tables, "keys": spec["keys"], "n_fields": len(spec["fields"]),
            "outcome_fields_ignored": spec["outcome_fields_ignored"]}


@app.get("/api/crash/examples")
def list_crash_examples():
    return [{"id": e["id"], "label": e["label"], "casenum": e["casenum"], "year": e["year"]}
            for e in crash_examples().values()]


@app.get("/api/crash/examples/{example_id}")
def get_crash_example(example_id: str):
    e = crash_examples().get(example_id)
    if e is None:
        raise HTTPException(404, "Unknown example")
    return {k: e[k] for k in ("id", "label", "casenum", "year", "record", "display", "recorded_outcome")}


@app.post("/api/crash/assess")
def assess_crash(body: CrashRecord):
    spec = crash_spec()
    record = crash_features.normalise(body.record)
    if len(record.get("accident", [])) > 1:
        raise HTTPException(422, "Send one crash at a time: the accident table must have one row.")
    cases = {r.get("CASENUM") for rows in record.values() for r in rows if r.get("CASENUM") is not None}
    if len(cases) > 1:
        raise HTTPException(422, f"Rows from more than one crash (CASENUM {sorted(map(str, cases))}); send one crash at a time.")
    needed = set(spec["tables"])
    if not needed & set(record):
        raise HTTPException(422, f"No required CRSS tables found; expected some of {sorted(needed)}.")

    outcome = set(spec["outcome_fields_ignored"])
    ignored = sorted({f for rows in record.values() for r in rows for f in r if f in outcome})
    missing = crash_features.missing_fields(record, spec)
    titles = spec["field_titles"]
    out = {}
    for target, label in [("y_injury", "injury_crash"), ("y_serious", "serious_or_fatal_crash")]:
        b = crash_model(target)
        feats = crash_features.build_features(record, b["features"], spec)
        X = pd.DataFrame([feats])[b["features"]].astype(float)
        predict = lambda frame: b["pipeline"].predict_proba(frame)[:, 1]
        p = float(predict(X)[0])
        # Main factors: change in probability when one field is treated as unknown
        # (the model then uses the training median, or no category, for that field's features).
        by_field = {}
        for f in b["features"]:
            t, fld, _, _ = crash_features.parse_feature(f)
            by_field.setdefault(f"{t}.{fld or 'n_rows'}", []).append(f)
        present = [k for k in by_field if k not in missing]
        if present:
            variants = pd.concat([X] * len(present), ignore_index=True)
            for i, k in enumerate(present):
                variants.loc[i, by_field[k]] = math.nan
            effects = p - predict(variants)
            factors = sorted(({"field": k, "label": titles[k], "effect": round(float(e), 4)}
                              for k, e in zip(present, effects)), key=lambda f: -abs(f["effect"]))[:6]
        else:
            factors = []
        out[label] = {"probability": round(p, 4), "threshold": round(float(b["threshold"]), 4),
                      "flagged": p >= b["threshold"], "model": b["model"],
                      "n_features": len(b["features"]), "n_fields": len(b["fields"]),
                      "factors": factors, "note": CRASH_NOTE}
    out["missing_fields"] = [{"field": k, "label": titles[k]} for k in missing]
    out["warning"] = (f"{len(missing)} of {len(spec['fields'])} required fields were not supplied and were treated as "
                      "unknown; confidence is reduced." if missing else None)
    out["ignored_outcome_fields"] = ignored
    out["disclaimer"] = DISCLAIMER
    return out


@app.get("/api/vehicles/makes")
def makes():
    p = profiles()
    return sorted(p["make"].dropna().unique().tolist())


@app.get("/api/vehicles/models")
def models(make: str):
    p = profiles()
    return sorted(p.loc[p["make"] == make, "model"].dropna().unique().tolist())


@app.get("/api/vehicles/years")
def years(make: str, model: str):
    p = profiles()
    return sorted(p.loc[(p["make"] == make) & (p["model"] == model), "model_year"].astype(int).unique().tolist(), reverse=True)


@app.get("/api/vehicles/profile")
def vehicle_profile(make: str, model: str, year: int = Query(..., ge=1950, le=2030)):
    p = profiles()
    hit = p[(p["make"] == make) & (p["model"] == model) & (p["model_year"] == year)]
    if hit.empty:
        raise HTTPException(404, "No complaints recorded for this vehicle in 2020-2024")
    r = hit.iloc[0]
    comp = {c[3:]: int(r[c]) for c in p.columns if c.startswith("c__") and r[c] > 0}
    nz = lambda v: None if pd.isna(v) else round(float(v), 4)
    return {
        "vehicle": {"make": make, "model": model, "model_year": year},
        "complaints": {"total_2020_2024": int(r.complaints), "serious": int(r.serious_complaints),
                       "serious_share": nz(r.serious_share), "crash": int(r.crash_complaints),
                       "fire": int(r.fire_complaints), "injury": int(r.injury_complaints),
                       "by_component": dict(sorted(comp.items(), key=lambda kv: -kv[1]))},
        "crash_exposure": {"crss_sample_vehicles": None if pd.isna(r.crss_sample_vehicles) else int(r.crss_sample_vehicles),
                           "estimated_crash_involved_vehicles": nz(r.est_crash_involved),
                           "injury_crash_rate": nz(r.crash_injury_rate), "serious_crash_rate": nz(r.crash_serious_rate),
                           "defect_recorded_rate": nz(r.crash_defect_rate)},
        "integrated": {"complaints_per_10k_crash_involved": nz(r.complaints_per_10k_crash_involved),
                       "percentile_among_vehicles": nz(r.complaint_rate_percentile),
                       "reliable": not pd.isna(r.complaints_per_10k_crash_involved)},
        "disclaimer": DISCLAIMER,
    }


@app.get("/api/vehicles/top")
def top_vehicles(limit: int = Query(25, le=100), min_complaints: int = 50):
    p = profiles()
    p = p[(p.complaints >= min_complaints) & p.complaints_per_10k_crash_involved.notna()]
    p = p.sort_values("complaints_per_10k_crash_involved", ascending=False).head(limit)
    return [{"make": r.make, "model": r.model, "model_year": int(r.model_year), "complaints": int(r.complaints),
             "complaints_per_10k_crash_involved": round(float(r.complaints_per_10k_crash_involved), 1),
             "serious_share": round(float(r.serious_share), 4)} for r in p.itertuples()]
