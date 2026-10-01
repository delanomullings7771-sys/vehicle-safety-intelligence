"""End-to-end tests for the deployed API, run against the exported artifacts.

    .venv/Scripts/python.exe -m pytest tests -q
"""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app" / "api"))
from main import app  # noqa: E402

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


def test_crash_form_has_labelled_options():
    form = client.get("/api/crash/form").json()
    selects = [f for f in form["fields"] if f["type"] == "select"]
    assert len(form["fields"]) == 30 and selects
    labels = [o["label"] for f in selects for o in f["options"]]
    assert "nan" not in [l.lower() for l in labels]


def test_crash_assessment_orders_risk_sensibly():
    form = {f["name"]: f for f in client.get("/api/crash/form").json()["fields"]}
    code = lambda field, text: next(o["value"] for o in form[field]["options"] if o["label"] == text)
    severe = {"first_harmful_event": code("first_harmful_event", "Rollover/Overturn"), "max_travel_speed": 75,
              "speed_limit": 65, "rollovers": 1, "unrestrained_occupants": 1, "vehicles": 1, "ejected": 1}
    minor = {"first_harmful_event": code("first_harmful_event", "Motor Vehicle In-Transport"),
             "manner_of_collision": code("manner_of_collision", "Front-to-Rear"), "max_travel_speed": 5,
             "speed_limit": 25, "vehicles": 2, "unrestrained_occupants": 0, "rollovers": 0}
    hi = client.post("/api/crash/assess", json=severe).json()
    lo = client.post("/api/crash/assess", json=minor).json()
    for key in ("injury_crash", "serious_or_fatal_crash"):
        assert hi[key]["probability"] > lo[key]["probability"]
    assert hi["serious_or_fatal_crash"]["flagged"] and not lo["serious_or_fatal_crash"]["flagged"]
    assert hi["warning"] and hi["injury_crash"]["factors"]


def test_crash_assessment_with_no_inputs_warns():
    r = client.post("/api/crash/assess", json={}).json()
    assert len(r["missing_inputs"]) == 30 and r["warning"]


def test_crash_rejects_unknown_field():
    assert client.post("/api/crash/assess", json={"not_a_field": 1}).status_code == 422


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
