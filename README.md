# Vehicle Safety Intelligence

MSc Applied Data Science capstone (COMP6830). One decision-support system built on two real, fully documented NHTSA sources, following CRISP-DM:

| Source | Type | Size |
|---|---|---|
| Crash Report Sampling System (CRSS) 2020-2024 | Structured, 28 relational tables per year | 264,661 crashes, about 3.6 GB |
| Vehicle Owner Complaints (FLAT_CMPL) | Structured metadata and unstructured narratives | 2,250,304 rows / 1,622,118 complaints, 1.6 GB |

The two sources are linked at **make + model + model year** (80% of CRSS crash vehicles match a complaint vehicle). The link puts complaint counts on an exposure-adjusted footing that neither source can provide alone.

## Models (selected on validation, scored once on untouched test data)

| | Task | Data | Selected model | Test result |
|---|---|---|---|---|
| S1 | Injury vs property-damage-only crash | CRSS | Gradient Boosting (1,317 fields from all 28 tables) | ROC-AUC 0.906, PR-AUC 0.918 (baseline 0.513) |
| S2 | Serious/fatal (K+A) crash | CRSS | Gradient Boosting | ROC-AUC 0.902, PR-AUC 0.642 (baseline 0.130) |
| U1 | Narrative to component groups (multi-label, 19) | Complaints | Linear SVM, TF-IDF | Top-1 accuracy 84.4%, micro-F1 0.740 |
| U2 | Narrative reports crash/fire/injury/death | Complaints | Linear SVM, TF-IDF | ROC-AUC 0.965, PR-AUC 0.858 (baseline 0.056) |

Each task compares a dummy baseline with 3-4 model families: Logistic Regression, Decision Tree, Random Forest and Gradient Boosting for S1/S2; Complement Naive Bayes, Logistic Regression and Linear SVM for U1/U2. Overfitting is checked with train-validation gaps and learning curves. Full results are in `outputs/evaluation/`.

## Splits
* CRSS: train 2020-2022, validate 2023, test 2024.
* Complaints: train 2010-2021, validate 2022-2023, test 2024-2026. Exact-duplicate narratives are removed so no test narrative repeats a training one.

## Reproduce

```
.venv/Scripts/python.exe src/audit_raw_data.py          # raw inventory and hashes
.venv/Scripts/python.exe src/build_crss_features.py     # crash-level table from all 28 tables
.venv/Scripts/python.exe src/build_complaints.py        # one row per complaint, harmonised components
.venv/Scripts/python.exe src/eda_crss.py                # EDA and statistical tests (CRSS)
.venv/Scripts/python.exe src/eda_complaints.py          # EDA, tests and exposure linkage (complaints)
.venv/Scripts/python.exe src/build_vehicle_profiles.py  # integrated make/model/year profiles
.venv/Scripts/python.exe src/train_crss.py              # S1, S2
.venv/Scripts/python.exe src/train_text.py              # U1, U2
.venv/Scripts/python.exe src/train_crash_deploy.py      # compact 30-input crash models for the app
.venv/Scripts/python.exe src/export_artifacts.py        # copy deployment artifacts to app/api/artifacts
.venv/Scripts/python.exe -m pytest tests -q             # end-to-end API tests
```

## Run the application locally

```
.venv/Scripts/python.exe -m uvicorn main:app --app-dir app/api --port 8000
.venv/Scripts/python.exe -m http.server 5500 --directory app/web
```
Then open http://localhost:5500.

## Deployment
* **API (Render):** `render.yaml` defines the service (root `app/api`). After deploying, set `ALLOWED_ORIGINS` to the Vercel URL.
* **Web (Vercel):** deploy `app/web` as a static site and set `window.API_BASE` in `app/web/config.js` to the Render URL.

## Restrictions
* CRSS is a probability sample: weighted national estimates are reported separately from sample-based model probabilities.
* Complaints are voluntary allegations, not verified defects. Exposure-adjusted rates are screening signals, not defect rates.
* Identifiers, sampling-design fields and outcome-encoding fields are excluded from predictors (`outputs/tables/crss_excluded_fields.csv`).
* All outputs support human review and do not establish causation, liability or the existence of a defect.
