# Vehicle Safety Intelligence

**Linking Structured Crash Data and Unstructured Owner Complaints to Support Vehicle Safety Screening**

MSc Applied Data Science capstone (COMP6830). One decision-support system built on two real, fully documented NHTSA sources, following CRISP-DM:

| Source | Type | Size |
|---|---|---|
| Crash Report Sampling System (CRSS) 2020-2024 | Structured, 28 relational tables per year | 264,661 crashes, about 3.6 GB |
| Vehicle Owner Complaints (FLAT_CMPL) | Structured metadata and unstructured narratives | 2,250,304 rows / 1,622,118 complaints, 1.6 GB |

The two sources are linked at **make + model + model year** (80% of CRSS crash vehicles match a complaint vehicle). The link sets complaint counts against crash involvement estimated from CRSS (a proxy for exposure), which neither source can provide alone.

## Models (chosen on validation data; test results calculated after every decision was fixed)

| | Task | Data | Final model | Test result |
|---|---|---|---|---|
| S1 | Injury vs property-damage-only crash | CRSS | Gradient Boosting, 15 features from 13 fields | PR-AUC 0.894 (baseline 0.513), ROC-AUC 0.872 |
| S2 | Serious/fatal (K+A) crash | CRSS | Gradient Boosting, 75 features from 47 fields | PR-AUC 0.610 (baseline 0.130), ROC-AUC 0.889 |
| U1 | Narrative to component groups (multi-label, 19) | Complaints | Linear SVM, TF-IDF | Top-1 accuracy 84.4% (baseline 21.1%), macro-F1 0.691 |
| U2 | Narrative reports crash/fire/injury/death | Complaints | Linear SVM, TF-IDF | PR-AUC 0.858 (baseline 0.064), ROC-AUC 0.965 |

Each task compares a dummy baseline with 3-4 model families: Logistic Regression, Decision Tree, Random Forest and Gradient Boosting for S1/S2; Complement Naive Bayes, Logistic Regression and Linear SVM for U1/U2.

**Crash field selection (CRISP-DM loop back to Data Preparation).** The benchmark crash models use all 303 documented fields (1,456 features). The fields were then reduced on evidence: VIN-decoded specifications removed after an ablation test (no gain), post-crash fields removed (recorded after the crash), and permutation-importance selection keeping the smallest feature set with at least 98% of the revised benchmark's validation PR-AUC. The final models need **50 CRSS fields from 12 tables**. Details: notebook 04 (Part B) and `outputs/selection/`.

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
.venv/Scripts/python.exe src/train_crss.py              # benchmark crash models (all fields)
.venv/Scripts/python.exe src/select_crash_features.py   # field selection: ablation, post-crash, 98% rule
.venv/Scripts/python.exe src/train_crss_final.py        # final S1, S2 on the selected features
.venv/Scripts/python.exe src/train_text.py              # U1, U2
.venv/Scripts/python.exe src/build_data_dictionary.py   # documentation/Data_Dictionary.xlsx
.venv/Scripts/python.exe src/export_artifacts.py        # deployment artifacts; checks the crash feature builder
.venv/Scripts/python.exe -m pytest tests -q             # end-to-end API tests
```

## Run the application locally

```
.venv/Scripts/python.exe -m uvicorn main:app --app-dir app/api --port 8000
.venv/Scripts/python.exe -m http.server 5500 --directory app/web
```
Then open http://localhost:5500.

The crash tool scores a crash record in CRSS format (`POST /api/crash/assess` with `{"record": {table: [rows]}}`; required fields at `GET /api/crash/schema`). The demonstration uses six real 2024 crashes (`GET /api/crash/examples`).

## Deployment
* **API (Render):** `render.yaml` defines the service (root `app/api`). After deploying, set `ALLOWED_ORIGINS` to the Vercel URL.
* **Web (Vercel):** deploy `app/web` as a static site and set `window.API_BASE` in `app/web/config.js` to the Render URL.

## Restrictions
* CRSS is a probability sample: weighted national estimates are reported separately from sample-based model probabilities.
* The crash models are retrospective: they analyse what was recorded about a crash (including some crash-event and investigation fields), not severity before a crash.
* Complaints are voluntary allegations, not verified defects. Rates per crash-involved vehicle are screening signals, not defect rates.
* Identifiers, sampling-design fields and outcome-encoding fields are excluded from predictors (`outputs/tables/crss_excluded_fields.csv`).
* All outputs support human review and do not establish causation, liability or the existence of a defect.
