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

### 1. Environment
Python 3.12 (developed on 3.12.14, Windows 11, 12 cores, about 15 GB RAM).

```
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

### 2. Get the raw data (about 6 GB unzipped; not in the repository)
Download from NHTSA and unzip into `data/raw/` with this layout (the scripts look for exactly these paths):

| File | Download | Unzip to |
|---|---|---|
| CRSS 2020-2024 (one zip per year) | `https://static.nhtsa.gov/nhtsa/downloads/CRSS/<year>/CRSS<year>CSV.zip` | `data/raw/crss/<year>/CRSS<year>CSV/` (28 CSV files, e.g. `accident.csv`) |
| Vehicle Owner Complaints (flat file) | `https://static.nhtsa.gov/odi/ffdd/cmpl/FLAT_CMPL.zip` | `data/raw/complaints/FLAT_CMPL/FLAT_CMPL.txt` |

* **CRSS is fixed:** the five zips used here are byte-for-byte the files NHTSA serves today (checked by size on 8 Oct 2026), so the crash results reproduce exactly.
* **The complaint file grows:** NHTSA adds complaints continuously. This project used the copy downloaded on 29 Sep 2026 (last complaint received 24 Sep 2026; 2,250,304 rows; SHA-256 `ca2d3fe0c53a7b2ecc5a5354162914ae21e046d7766452a1ee8825ea0ceb4ad1`). A newer download adds complaints to the 2024-2026 test period, so the text-model test results will differ slightly; training and validation (to 2023) are unchanged. `src/audit_raw_data.py` prints the hash of the file you have.

### 3. Run the pipeline in this order
Total about 4.5 hours, most of it `train_crss.py` (about 3 hours).

```
.venv/Scripts/python.exe src/audit_raw_data.py          # raw inventory and hashes
.venv/Scripts/python.exe src/build_crss_features.py     # crash-level table from all 28 tables (about 4 min)
.venv/Scripts/python.exe src/build_complaints.py        # one row per complaint, harmonised components (about 15 min)
.venv/Scripts/python.exe src/build_data_dictionary.py   # first pass: field roles (field selection reads them)
.venv/Scripts/python.exe src/eda_crss.py                # EDA and statistical tests (CRSS)
.venv/Scripts/python.exe src/eda_complaints.py          # EDA, tests and exposure linkage (complaints)
.venv/Scripts/python.exe src/build_vehicle_profiles.py  # integrated make/model/year profiles
.venv/Scripts/python.exe src/train_crss.py              # benchmark crash models (all fields)
.venv/Scripts/python.exe feature_reduction/feature_reduction.py  # reduction study (notebook 05b)
.venv/Scripts/python.exe src/select_crash_features.py   # field selection: ablation, post-crash, 98% rule
.venv/Scripts/python.exe src/train_crss_final.py        # final S1, S2 on the selected features
.venv/Scripts/python.exe src/train_text.py              # U1, U2
.venv/Scripts/python.exe src/build_data_dictionary.py   # second pass: adds each field's selection status
.venv/Scripts/python.exe src/export_artifacts.py        # deployment artifacts; checks the crash feature builder
.venv/Scripts/python.exe -m pytest tests -q             # end-to-end API tests
```

All random steps use a fixed seed (42), so a rerun gives the same results.

### 4. Notebooks
Notebooks 01-06 and 05b (in `notebooks/`) read the saved outputs, so they can be opened and re-run without retraining once the steps above (or just the data steps) have run. To execute them all:

```
.venv/Scripts/jupyter-nbconvert.exe --to notebook --execute --inplace notebooks/*.ipynb
```

Without the raw data, notebooks 01 and 06 and the API tests still run from the committed outputs and `app/api/artifacts/`.

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
