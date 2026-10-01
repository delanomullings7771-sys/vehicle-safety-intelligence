"""Vehicle Safety Intelligence API.

Serves the trained models as three human-review workflows:
  /api/complaint/analyze   U1 component routing + U2 serious-incident flag for a narrative
  /api/crash/assess        S1 injury + S2 serious/fatal likelihood for a crash description
  /api/vehicles/...        integrated make/model/year profile (complaints + CRSS exposure/severity)
Artifacts are produced by src/export_artifacts.py into app/api/artifacts.
"""
import json
import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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
def crash_form() -> dict:
    return json.loads((ART / "crash_form.json").read_text(encoding="utf-8"))


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


@app.get("/api/crash/form")
def get_crash_form():
    return crash_form()


@app.post("/api/crash/assess")
def assess_crash(inputs: dict[str, float | int | None]):
    form = crash_form()
    known = {f["name"] for f in form["fields"]}
    unknown = sorted(set(inputs) - known)
    if unknown:
        raise HTTPException(422, f"Unknown fields: {unknown}")
    row, missing = {}, []
    for f in form["fields"]:
        v = inputs.get(f["name"])
        if v is None:
            missing.append(f["label"])
        row[f["name"]] = np.nan if v is None else float(v)
    X = pd.DataFrame([row])
    provided = [f["name"] for f in form["fields"] if inputs.get(f["name"]) is not None]
    labels = {f["name"]: f["label"] for f in form["fields"]}
    out = {}
    for target, label in [("y_injury", "injury_crash"), ("y_serious", "serious_or_fatal_crash")]:
        b = crash_model(target)
        predict = lambda frame: b["pipeline"].predict_proba(frame[b["features"]])[:, 1]
        p = float(predict(X)[0])
        # Local explanation: change in probability when each provided input is set back to unknown.
        if provided:
            variants = pd.concat([X] * len(provided), ignore_index=True)
            for i, name in enumerate(provided):
                variants.loc[i, name] = np.nan
            effects = p - predict(variants)
            factors = sorted(({"label": labels[n], "effect": round(float(e), 4)} for n, e in zip(provided, effects)),
                             key=lambda f: -abs(f["effect"]))[:6]
        else:
            factors = []
        out[label] = {"probability": round(p, 4), "threshold": round(b["threshold"], 4),
                      "flagged": p >= b["threshold"], "model": b["model"], "factors": factors,
                      "note": "Probability is relative to the CRSS sample, which over-represents injury crashes."}
    out["missing_inputs"] = missing
    out["warning"] = (f"{len(missing)} inputs were not provided and were treated as unknown; confidence is reduced."
                      if missing else None)
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
