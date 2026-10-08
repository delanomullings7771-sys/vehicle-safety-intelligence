"""Build the project data dictionary (documentation/Data_Dictionary.xlsx).

Field titles and definitions come from official NHTSA documentation:
  * CRSS Analytical User's Manual 2016-2024 (documentation/crss/analytical_manual/813796_clean.pdf)
  * Complaint flat-file layout (documentation/complaints/CONSUMER COMPLAINTS.txt)
Each field's role in this project is taken from the build scripts' own registers, so the
dictionary always matches what the code actually did.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from pypdf import PdfReader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_complaints
import build_crss_features as crss
from project_paths import CRSS_YEARS, OUTPUT_DIR, PROCESSED_DIR, PROJECT_ROOT, TABLE_DIR, crss_extracted_dir

DOCS = PROJECT_ROOT / "documentation"
OUT = DOCS / "Data_Dictionary.xlsx"
MANUAL = DOCS / "crss" / "analytical_manual" / "813796_clean.pdf"
PAGE_HEADER = re.compile(r"Data Element Definitions and Codes[^\n]*\n\s*\d+\s*\nCrash Report Sampling System Analytical User.s Manual, 2016-2024\s*\n")
SAS_NAME = re.compile(r"\nSAS Name\s*\n([A-Z][A-Za-z0-9_]*)")
DEFINITION = re.compile(r"\n([^\n]+)\nDefinition\s*\n")


def crss_manual_definitions() -> pd.DataFrame:
    """Element ID, title and definition for every SAS name in the analytical manual.

    Each entry reads: [ID.] Title / Definition / text / Additional Information / ... / SAS Name / NAME.
    Starting from each SAS name, the nearest preceding "Definition" heading gives the title and text.
    """
    text = "\n".join(p.extract_text() or "" for p in PdfReader(MANUAL).pages)
    text = PAGE_HEADER.sub("\n", text.replace("�", "'"))
    defs = list(DEFINITION.finditer(text))
    rows = []
    for m in SAS_NAME.finditer(text):
        prior = [d for d in defs if d.end() < m.start()]
        if not prior:
            continue
        d = prior[-1]
        block = text[d.end(): m.start()]
        definition = re.sub(r"\s+", " ", re.split(r"\nAdditional Information", block)[0]).strip()
        title_line = d.group(1).strip()
        idm = re.match(r"^([A-Z]{1,3}\d+[A-Z]*)\.\s+(.*)$", title_line)
        rows.append({"field": m.group(1).upper(), "element_id": idm.group(1) if idm else "",
                     "title": idm.group(2) if idm else title_line, "definition": definition})
    # Fallback: the manual's data-element list ("ID  Name  SAS NAME" per line) gives an official
    # title for elements whose full entry is filed under a related name.
    listed = re.findall(r"\n([A-Z]{1,3}\d+[A-Z]*(?:/[A-Z]+\d+)?)\s+([^\n]+?)\s+([A-Z][A-Z0-9_]{2,})\s*(?=\n)", text)
    have = {r["field"] for r in rows}
    for element_id, title, sas in listed:
        if sas not in have:
            rows.append({"field": sas, "element_id": element_id, "title": title.strip(),
                         "definition": f"Official element: {title.strip()} (see manual element {element_id})."})
            have.add(sas)
    return pd.DataFrame(rows).drop_duplicates("field")


def complaint_layout() -> pd.DataFrame:
    """Parse the 51-field layout from the official complaint file description."""
    text = (DOCS / "complaints" / "CONSUMER COMPLAINTS.txt").read_text(encoding="latin-1")
    text = text[text.index("Field#"):]
    rows, current = [], None
    for line in text.splitlines():
        m = re.match(r"^(\d{1,2})\s+([A-Z_]+)\s+(\S+\(\d+\))\s+(.*)$", line)
        if m:
            current = {"field_no": int(m.group(1)), "field": m.group(2), "type": m.group(3),
                       "official_description": m.group(4).strip()}
            rows.append(current)
        elif current and line.startswith(" " * 20) and line.strip():
            current["official_description"] += " " + line.strip()
    df = pd.DataFrame(rows)
    df["official_description"] = df.official_description.str.replace(r"\s+", " ", regex=True)
    return df


def complaint_role(field: str) -> str:
    roles = {
        "CDESCR": "MODEL INPUT: narrative text for U1 and U2 (cleaned of editorial markers)",
        "COMPDESC": "U1 TARGET: harmonised into 19 component groups (not a predictor)",
        "CRASH": "U2 TARGET: part of the serious-incident label (not a predictor)",
        "FIRE": "U2 TARGET: part of the serious-incident label (not a predictor)",
        "INJURED": "U2 TARGET: part of the serious-incident label (not a predictor)",
        "DEATHS": "U2 TARGET: part of the serious-incident label (not a predictor)",
        "ODINO": "KEY: groups repeated component rows into one complaint",
        "LDATE": "SPLIT: date received, used for the chronological train/validation/test split",
        "DATEA": "Checked against LDATE; not used further",
        "PROD_TYPE": "FILTER: only vehicle complaints (V) are modelled",
        "MAKETXT": "LINKAGE KEY: make (vehicle profiles; app lookup)",
        "MODELTXT": "LINKAGE KEY: model (vehicle profiles; app lookup)",
        "YEARTXT": "LINKAGE KEY: model year (vehicle profiles; app lookup)",
        "STATE": "Kept in prepared table for reference; not a model input",
    }
    if field in roles:
        return roles[field]
    if field in {"CMPLID", "VIN", "CITY", "DEALER_NAME", "DEALER_TEL", "DEALER_CITY", "DEALER_STATE", "DEALER_ZIP"}:
        return "NOT USED: identifier or personal/contact detail"
    return "NOT USED in models: structured detail, often blank; narrative carries the information"


def crss_raw_fields(defs: pd.DataFrame) -> pd.DataFrame:
    reg = pd.read_csv(TABLE_DIR / "crss_feature_register.csv")
    excluded = pd.read_csv(TABLE_DIR / "crss_excluded_fields.csv")
    used = set(zip(reg.table, reg.field))
    excl = {(r.table, r.field): r.reason for r in excluded.itertuples()}
    imputed_pairs = {"HOUR": "HOUR_IM", "DAY_WEEK": "WKDY_IM", "HARM_EV": "EVENT1_IM", "MAN_COLL": "MANCOL_IM",
                     "RELJCT1": "RELJCT1_IM", "RELJCT2": "RELJCT2_IM", "LGT_COND": "LGTCON_IM",
                     "WEATHER": "WEATHR_IM", "ALCOHOL": "ALCHL_IM", "MOD_YEAR": "MDLYR_IM", "IMPACT1": "IMPACT1_IM",
                     "M_HARM": "VEVENT_IM", "VEH_ALCH": "V_ALCH_IM", "P_CRASH1": "PCRASH1_IM", "AGE": "AGE_IM",
                     "SEX": "SEX_IM", "SEAT_POS": "SEAT_IM", "EJECTION": "EJECT_IM", "DRINKING": "PERALCH_IM",
                     "BODY_TYP": "BDYTYP_IM"}
    rows = []
    for table in crss.TABLE_LEVEL:
        years = {}
        for y in CRSS_YEARS:
            for c in pd.read_csv(crss_extracted_dir(y) / f"{table}.csv", nrows=0, encoding="latin-1").columns:
                years.setdefault(c.upper(), []).append(y)
        for field, ys in years.items():
            key = (table, field)
            if key in excl:
                reason = excl[key]
                role = ("EXCLUDED - leakage: " if field in crss.LEAKAGE else "EXCLUDED: ") + reason
            elif key in used:
                role = "USED: " + ("crash-level predictor" if table == "accident" else "aggregated to crash level")
                if field in crss.POST_CRASH or table == "damage":
                    role += " (tagged post-crash)"
            elif field.endswith("NAME") and field[:-4] in years:
                role = "Text label of the code in " + field[:-4] + " (code used instead)"
            elif field in imputed_pairs and imputed_pairs[field] in years:
                role = f"Replaced by NHTSA's imputed version {imputed_pairs[field]}"
            elif table != "accident" and field in crss.REPEATED:
                role = "Repeated from the accident table (used there)"
            elif field == "YEAR":
                role = "Added by project: data year (used for the split)"
            else:
                role = "Not used: constant or not applicable"
            rows.append({"table": table, "grain": crss.TABLE_LEVEL[table], "field": field,
                         "years_present": "all" if len(ys) == len(CRSS_YEARS) else ", ".join(map(str, ys)),
                         "project_role": role})
    df = pd.DataFrame(rows).merge(defs, on="field", how="left")
    vpic = df.table.str.startswith("vpic")
    original = {}
    for t in ("vpicdecode", "vpictrailerdecode"):
        for c in pd.read_csv(crss_extracted_dir(2024) / f"{t}.csv", nrows=0, encoding="latin-1").columns:
            original[c.upper()] = c
    def vpic_title(field):
        name = original.get(field, field)
        name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|_", " ", name).strip()
        return re.sub(r"\s+Id$|\s+ID$", " (code)", name)
    df.loc[vpic & df.title.isna(), "title"] = df.loc[vpic & df.title.isna(), "field"].map(vpic_title)
    df.loc[df.field == "GVWR_TO", ["element_id", "title", "definition"]] = [
        "V18", "Power Unit Gross Vehicle Weight Rating (GVWR), upper bound",
        "Upper end of the gross vehicle weight rating class of the power unit (element V18; GVWR_FROM gives the lower end)."]
    df.loc[df.field == "PGVWR_TO", ["title", "definition"]] = [
        "Parked vehicle GVWR, upper bound", "Upper end of the gross vehicle weight rating class of the parked/working vehicle."]
    df.loc[vpic & df.definition.isna(), "definition"] = (
        "Manufacturer specification decoded from the vehicle's VIN by NHTSA's vPIC service "
        "(field names follow the vPIC data dictionary).")
    return df[["table", "grain", "field", "element_id", "title", "definition", "years_present", "project_role"]]


def crss_feature_list(raw: pd.DataFrame) -> pd.DataFrame:
    import eda_crss
    labels = eda_crss.code_labels()
    reg = pd.read_csv(TABLE_DIR / "crss_feature_register.csv")
    title = raw.drop_duplicates("field").set_index("field")["title"]
    def explain(r):
        rest = r.feature.split("__", 1)[1]
        if r.kind == "categorical_code":
            return f"Crash-level code of {r.field} (one-hot encoded in the models)"
        if "=" in rest:
            code = rest.split("=")[1].removesuffix(".0")
            lab = labels.get((r.table, r.field, code))
            return f"Number of {r.grain}s in the crash with {r.field} = {code}" + (f" ({lab})" if lab else "")
        if rest.endswith("__min") or rest.endswith("__max"):
            return f"{'Lowest' if rest.endswith('__min') else 'Highest'} {r.field} among the crash's {r.grain}s"
        if rest == "n_rows":
            return f"Number of {r.table} records in the crash"
        return f"Crash-level value of {r.field}"
    reg["field_title"] = reg.field.map(title)
    reg["meaning"] = reg.apply(explain, axis=1)
    return reg[["feature", "table", "grain", "field", "field_title", "kind", "group", "meaning"]]


def selection(raw: pd.DataFrame, features: pd.DataFrame) -> tuple:
    """Field selection (notebook 04, Part B): what happened to each used field, and the 50 final-model fields."""
    sel_dir = OUTPUT_DIR / "selection"
    if not (sel_dir / "crss_selected_features.json").exists():
        # First build, before select_crash_features.py has run (it reads the field roles from this dictionary).
        print("No field selection yet: selection columns left empty; rerun after select_crash_features.py")
        return (raw.assign(selection_status=""), features.assign(source="", in_final_model=""),
                pd.DataFrame(columns=["step", "fields", "note"]),
                pd.DataFrame(columns=["table", "field", "used_by", "title", "definition", "features_built"]))
    sel = json.loads((sel_dir / "crss_selected_features.json").read_text())
    s1, s2 = set(sel["y_injury"]["fields"]), set(sel["y_serious"]["fields"])
    f1, f2 = set(sel["y_injury"]["features"]), set(sel["y_serious"]["features"])

    def models(key, a, b):
        return "S1 and S2" if key in a and key in b else ("S1" if key in a else ("S2" if key in b else ""))

    key = raw.table + "." + raw.field
    vin = raw.table.str.startswith("vpic") | raw.field.str.upper().str.startswith("VPIC")
    used = raw.project_role.str.startswith("USED")
    post = raw.project_role.str.contains("post-crash")
    final = key.map(lambda k: models(k, s1, s2))
    raw = raw.assign(selection_status=np.select(
        [final != "", used & vin, used & post, used],
        ["FINAL MODEL INPUT (" + final + ")", "Removed in field selection: VIN-decoded (ablation showed no gain)",
         "Removed in field selection: post-crash (recorded after the crash)",
         "Not selected: below the 98% importance rule"], ""))
    features = features.assign(
        source=np.select([features.table.str.startswith("vpic") | features.field.str.upper().str.startswith("VPIC"),
                          features.group == "post_crash"], ["VIN-decoded", "post-crash"], "police-reported"),
        in_final_model=features.feature.map(lambda f: models(f, f1, f2)))
    funnel = pd.read_csv(sel_dir / "crss_field_funnel.csv").fillna("")
    titles = raw.drop_duplicates(["table", "field"]).set_index(["table", "field"])
    fin = pd.DataFrame([k.split(".") for k in sel["deployment_fields"]], columns=["table", "field"])
    fin["used_by"] = [models(k, s1, s2) for k in sel["deployment_fields"]]
    fin["title"] = [("Number of rows in this table for the crash" if f == "n_rows" else titles.title.get((t, f)))
                    for t, f in zip(fin.table, fin.field)]
    fin["definition"] = [None if f == "n_rows" else titles.definition.get((t, f)) for t, f in zip(fin.table, fin.field)]
    prefix = lambda t: "acc" if t == "accident" else t
    fin["features_built"] = [", ".join(sorted(x for x in f1 | f2 if x.split("__")[0] == prefix(t)
                                              and x.split("__")[1].split("=")[0] == f))
                             for t, f in zip(fin.table, fin.field)]
    return raw, features, funnel, fin


def write(sheets: dict) -> None:
    with pd.ExcelWriter(OUT, engine="openpyxl") as xl:
        for name, df in sheets.items():
            df.to_excel(xl, sheet_name=name, index=False)
            ws = xl.sheets[name]
            ws.freeze_panes = "A2"
            for cell in ws[1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="1F4E79")
                cell.alignment = Alignment(wrap_text=True, vertical="top")
            for i, col in enumerate(df.columns, start=1):
                q = df[col].astype(str).str.len().quantile(0.9) if len(df) else 0
                width = min(70, max(12, int(q) + 2, len(str(col)) + 2))
                ws.column_dimensions[get_column_letter(i)].width = width
                if width >= 40:
                    for cell in ws[get_column_letter(i)][1:]:
                        cell.alignment = Alignment(wrap_text=True, vertical="top")
            if len(df) > 1:
                ws.auto_filter.ref = ws.dimensions


def main() -> None:
    defs = crss_manual_definitions()
    raw = crss_raw_fields(defs)
    features = crss_feature_list(raw)
    raw, features, funnel, final_fields = selection(raw, features)
    layout = complaint_layout()
    layout["project_role"] = layout.field.map(complaint_role)
    groups = [g for g in build_complaints.GROUPS]
    prepared = pd.DataFrame([
        ("ODINO", "Complaint reference number (one row per complaint)"),
        ("date / year", "Date and year NHTSA received the complaint (LDATE)"),
        ("split", "train (to 2021), validation (2022-2023) or test (2024 onward), by date received"),
        ("make / model / model_year", "Vehicle as reported (MAKETXT, MODELTXT, YEARTXT)"),
        ("state", "Complainant's state"),
        ("text", "Narrative (CDESCR) with editorial markers and FOIA redaction notices removed; the only model input"),
        ("components", "List of harmonised component groups named in the complaint"),
        ("crash / fire / injured / deaths", "Whether the complaint reports a crash, fire, any injury, any death"),
        ("y_serious", "U2 target: 1 if crash, fire, injury or death is reported"),
        ("n_rows", "Number of raw component rows combined into this complaint"),
    ] + [(f"c__{g}", f"U1 target: 1 if the complaint names component group {g}") for g in groups],
        columns=["column", "meaning"])
    targets = pd.DataFrame([
        ("S1", "y_injury", "CRSS crash", "1 if MAX_SEV is C, B, A, K or injured-severity-unknown; 0 if O (no apparent injury); excluded if unknown"),
        ("S2", "y_serious", "CRSS crash", "1 if MAX_SEV is A (suspected serious) or K (fatal); 0 if O, C or B; excluded if unknown"),
        ("U1", "c__<GROUP> (19 columns)", "Complaint", "Multi-label: the harmonised component groups in COMPDESC"),
        ("U2", "y_serious", "Complaint", "1 if CRASH = Y, FIRE = Y, INJURED > 0 or DEATHS > 0"),
    ], columns=["model", "column", "unit", "definition"])
    mapping = pd.read_csv(TABLE_DIR / "complaint_component_mapping.csv", keep_default_na=False)
    profiles = pd.read_parquet(PROCESSED_DIR / "vehicle_profiles.parquet")
    profile_meaning = {
        "mk": "Normalised make (letters and digits only), join key", "md": "Normalised model, join key",
        "model_year": "Model year, join key", "make": "Make as most often written in complaints",
        "model": "Model as most often written in complaints", "complaints": "Complaints received 2020-2024",
        "serious_complaints": "Of which report a crash, fire, injury or death",
        "crash_complaints": "Of which report a crash", "fire_complaints": "Of which report a fire",
        "injury_complaints": "Of which report an injury", "serious_share": "serious_complaints / complaints",
        "crss_sample_vehicles": "CRSS sampled vehicles of this make/model/year, 2020-2024",
        "est_crash_involved": "Survey-weighted national estimate of crash-involved vehicles, 2020-2024 (a proxy for exposure, not vehicles on the road)",
        "crash_injury_rate": "Weighted share of its crashes with injury (shown if at least 20 sampled vehicles)",
        "crash_serious_rate": "Weighted share of its crashes with serious or fatal injury",
        "crash_defect_rate": "Weighted share with a police-recorded vehicle defect",
        "crss_make_name": "Make as named in CRSS", "crss_model_name": "Model as named in CRSS",
        "complaints_per_10k_crash_involved": "Complaints received 2020-2024 per 10,000 estimated crash-involved vehicles",
        "complaint_rate_percentile": "Percentile of that rate among vehicles with at least 20 sampled CRSS vehicles",
    }
    prof = pd.DataFrame({"column": profiles.columns})
    prof["meaning"] = prof.column.map(profile_meaning).fillna(
        prof.column.str.replace("c__", "Complaints naming component group ", regex=False))
    about = pd.DataFrame({"item": [
        "Project", "Sources", "Sheets", "CRSS fields", "CRSS features", "Field funnel", "Final model fields",
        "Complaint fields", "Complaint model table", "Targets", "Component mapping", "Vehicle profiles", "Roles",
        "Selection status", "Regenerate"], "description": [
        "Vehicle Safety Intelligence: Linking Structured Crash Data and Unstructured Owner Complaints to Support Vehicle Safety Screening. MSc Applied Data Science capstone (COMP6830)",
        "NHTSA CRSS 2020-2024 (28 tables per year) and NHTSA Vehicle Owner Complaints (51 fields). Definitions are quoted from the CRSS Analytical User's Manual 2016-2024 and the official complaint file layout.",
        "One sheet per part of the data, described below.",
        f"Every raw field in all 28 CRSS tables ({len(raw):,} table/field pairs): official title and definition, years present, its role in data preparation and its status after field selection.",
        f"All {len(features):,} crash-level features built in data preparation, with the field and code each one counts or summarises, its source tag, and whether the final S1/S2 models use it.",
        "How the 1,114 raw CRSS fields become the 178 fields available to importance-based selection (notebook 04, Part B).",
        f"The {len(final_fields)} CRSS fields the final crash models use (S1 13, S2 47): what a crash record must contain for the deployed crash tool.",
        "All 51 complaint fields with official descriptions and their role in this project.",
        "Columns of the prepared one-row-per-complaint table used by the text models.",
        "Exact definitions of the four model targets (S1, S2, U1, U2).",
        "How the 55 NHTSA component names were grouped into 19 component groups.",
        "Columns of the make/model/year table linking complaints to CRSS exposure (used by the web app).",
        "project_role (data preparation): USED = used to build the benchmark features; EXCLUDED = removed with the stated reason (leakage, identifier, design, key, text label); Replaced = NHTSA's imputed version used instead.",
        "selection_status (field selection, notebook 04 Part B): FINAL MODEL INPUT, removed as VIN-decoded or post-crash, or not selected under the 98% rule.",
        "python src/build_data_dictionary.py (reads the build scripts' registers, so it always matches the code)."]})
    write({"About": about, "CRSS fields": raw, "CRSS features": features, "Field funnel": funnel,
           "Final model fields": final_fields,
           "Complaint fields": layout[["field_no", "field", "type", "official_description", "project_role"]],
           "Complaint model table": prepared, "Targets": targets, "Component mapping": mapping,
           "Vehicle profiles": prof})
    covered = raw.definition.notna().mean()
    print(f"written {OUT} | CRSS fields {len(raw):,} (definitions for {covered:.0%}) | features {len(features):,} | complaint fields {len(layout)}")
    print(raw.project_role.str.split(":").str[0].str.split(" -").str[0].value_counts().to_string())
    print(raw.selection_status.replace("", "(not used)").str.split(":").str[0].value_counts().to_string())
    print("final model fields:", len(final_fields))
    missing = raw[raw.definition.isna() & ~raw.field.str.endswith("NAME")]
    print("fields without a manual definition (excluding NAME labels):", len(missing))
    print(missing[["table", "field"]].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
