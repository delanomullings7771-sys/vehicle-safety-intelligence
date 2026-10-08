"""End-to-end tests for the deployed API, run against the exported artifacts.

    .venv/Scripts/python.exe -m pytest tests -q
"""
import copy
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app" / "api"))
from main import ART, app  # noqa: E402

client = TestClient(app)

AIRBAG_CRASH = ("I was rear-ended at a stop light and the air bags did not deploy. My head hit the steering wheel "
                "and I was taken to the hospital by ambulance. A police report was filed.")
BRAKES = ("When pressing the brake pedal the pedal goes to the floor and the vehicle takes a long distance to stop. "
          "The dealer replaced the brake master cylinder but the problem continues.")


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_meta_lists_four_models_with_test_metrics():
    meta = client.get("/api/meta").json()
    assert len(meta["models"]) == 4
    assert {m["task"][:2] for m in meta["models"]} == {"S1", "S2", "U1", "U2"}
    assert "human review" in meta["disclaimer"]


@pytest.mark.parametrize("text, component, serious", [(AIRBAG_CRASH, "AIR BAGS", True), (BRAKES, "BRAKES", False)])
def test_complaint_routing_and_flag(text, component, serious):
    r = client.post("/api/complaint/analyze", json={"text": text})
    assert r.status_code == 200
    body = r.json()
    recommended = [c["component"] for c in body["components"] if c["recommended"]]
    assert component in recommended
    assert body["serious_incident"]["flagged"] is serious
    assert body["components"][0]["evidence"], "recommendation should come with evidence terms"


def test_complaint_rejects_short_text():
    assert client.post("/api/complaint/analyze", json={"text": "too short"}).status_code == 422


def crash_example(i=0):
    first = client.get("/api/crash/examples").json()[i]["id"]
    return client.get(f"/api/crash/examples/{first}").json()


def test_crash_schema_lists_the_50_fields_from_12_tables():
    schema = client.get("/api/crash/schema").json()
    assert schema["n_fields"] == 50 and len(schema["tables"]) == 12
    assert "MAX_SEV" in schema["outcome_fields_ignored"]


def test_crash_examples_score_exactly_as_in_training():
    """Trained = evaluated = deployed: features built from the raw record give the model's training-data score."""
    stored = {e["id"]: e for e in json.loads((ART / "crash_examples.json").read_text())["examples"]}
    assert len(stored) >= 4
    for ex_id, e in stored.items():
        r = client.post("/api/crash/assess", json={"record": e["record"]}).json()
        assert r["missing_fields"] == [] and r["warning"] is None
        for target, key in [("y_injury", "injury_crash"), ("y_serious", "serious_or_fatal_crash")]:
            b = joblib.load(ART / f"crash_{target}.joblib")
            row = pd.DataFrame([e["training_features"]])[b["features"]].astype(float)
            expected = float(b["pipeline"].predict_proba(row)[:, 1][0])
            assert abs(r[key]["probability"] - expected) < 1e-4, (ex_id, key)
            assert r[key]["flagged"] == (r[key]["probability"] >= r[key]["threshold"])
            assert r[key]["factors"]


def test_crash_outcome_and_extra_fields_are_ignored():
    e = crash_example()
    base = client.post("/api/crash/assess", json={"record": e["record"]}).json()
    rec = copy.deepcopy(e["record"])
    rec["accident"][0].update({"MAX_SEV": 4, "NOT_A_MODEL_FIELD": 1})
    r = client.post("/api/crash/assess", json={"record": rec}).json()
    assert r["ignored_outcome_fields"] == ["MAX_SEV"]
    assert r["injury_crash"]["probability"] == base["injury_crash"]["probability"]


def test_crash_missing_table_is_flagged_not_filled():
    rec = {k: v for k, v in crash_example()["record"].items() if k != "person"}
    r = client.post("/api/crash/assess", json={"record": rec}).json()
    assert r["warning"] and len(r["missing_fields"]) == 14
    assert all(m["field"].startswith("person.") for m in r["missing_fields"])


def test_crash_rejects_rows_from_two_crashes():
    rec = copy.deepcopy(crash_example()["record"])
    rec["vehicle"][0]["CASENUM"] = 1
    assert client.post("/api/crash/assess", json={"record": rec}).status_code == 422
    assert client.post("/api/crash/assess", json={"record": {"weather": []}}).status_code == 422


def test_vehicle_lookup_chain():
    assert "FORD" in client.get("/api/vehicles/makes").json()
    assert "ESCAPE" in client.get("/api/vehicles/models", params={"make": "FORD"}).json()
    assert 2017 in client.get("/api/vehicles/years", params={"make": "FORD", "model": "ESCAPE"}).json()
    p = client.get("/api/vehicles/profile", params={"make": "FORD", "model": "ESCAPE", "year": 2017}).json()
    assert p["complaints"]["total_2020_2024"] > 0
    assert p["integrated"]["reliable"] and 0 < p["integrated"]["percentile_among_vehicles"] <= 1
    assert next(iter(p["complaints"]["by_component"])) == "ENGINE"


def test_vehicle_profile_missing_returns_404():
    r = client.get("/api/vehicles/profile", params={"make": "FORD", "model": "NOT A MODEL", "year": 2017})
    assert r.status_code == 404


def test_top_vehicles_sorted_by_exposure_adjusted_rate():
    top = client.get("/api/vehicles/top", params={"limit": 10}).json()
    rates = [t["complaints_per_10k_crash_involved"] for t in top]
    assert len(top) == 10 and rates == sorted(rates, reverse=True)
