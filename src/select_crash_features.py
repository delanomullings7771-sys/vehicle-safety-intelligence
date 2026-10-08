"""Data Preparation (loop back from Modelling): evidence-based selection of the crash fields (S1, S2).

The benchmark models (train_crss.py) use every documented field from all 28 CRSS tables (303 fields,
1,456 features). Evaluating them raised three questions, answered here in order (CRISP-DM loop back to
Data Preparation, task "select data"):

  1. Do the VIN-decoded specifications help?   Ablation: retrain without them and compare.
  2. Should post-crash fields stay?             They are recorded after the crash (damage, towing, alcohol tests)
                                                 and mostly reflect its outcome, so they are removed; the cost is measured.
  3. How few features are needed?               Rank the remaining features by permutation importance on
                                                 validation data, retrain on the top k, and keep the smallest k
                                                 that retains at least 98% of the benchmark's validation PR-AUC.

Every decision uses the training years (2020-2022) and validation year (2023) only. The test year (2024)
is reported for confirmation and never used to choose anything.

Outputs (outputs/selection/):
  crss_field_funnel.csv               how 1,114 raw fields become the final inputs
  crss_selection_ablation.csv            full-model results with and without VIN and post-crash fields
  crss_<target>_selection_importance.csv  permutation-importance ranking of the remaining features
  crss_<target>_selection_topk.csv        retrained top-k models (validation and test)
  crss_selected_features.json          the selected features and raw fields per target
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_crss
from project_paths import (CRSS_TEST_YEAR, CRSS_TRAIN_YEARS, CRSS_VALIDATION_YEAR, EVALUATION_DIR, OUTPUT_DIR,
                           PROJECT_ROOT, TABLE_DIR)

OUT = OUTPUT_DIR / "selection"
OUT.mkdir(parents=True, exist_ok=True)
RNG = 42
RULE = 0.98  # keep the smallest feature set retaining at least 98% of the benchmark's validation PR-AUC
KS = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300]


def is_vin(reg: pd.DataFrame) -> pd.Series:
    """VIN-decoded specifications: the two vPIC tables plus the vPIC body class copied into vehicle/person."""
    return reg.table.str.startswith("vpic") | reg.field.str.upper().str.startswith("VPIC")


def gb(params: dict) -> HistGradientBoostingClassifier:
    """Same configuration as the benchmark Gradient Boosting models."""
    return HistGradientBoostingClassifier(learning_rate=0.08, max_iter=600, l2_regularization=1.0,
                                          early_stopping=True, validation_fraction=0.1, n_iter_no_change=30,
                                          random_state=RNG, **params)


def benchmark_params(target: str) -> dict:
    sel = pd.read_csv(EVALUATION_DIR / f"crss_{target}_model_comparison.csv")
    sel = sel[sel.selected].iloc[0]
    return {k: v for k, v in json.loads(sel.params).items() if v is not None and k in ("max_leaf_nodes", "min_samples_leaf")}


def fit_score(feats, parts, reg, target, params):
    """Fit preprocessing on training years, train Gradient Boosting, score validation and test."""
    tr, va, te = parts
    feats = train_crss.drop_redundant(tr[feats])
    cat = [f for f in feats if reg.loc[f, "kind"] == "categorical_code"]
    prep = train_crss.preprocessor(cat, [f for f in feats if f not in cat]).fit(tr[feats])
    Xtr, Xva, Xte = (prep.transform(x[feats]).astype(np.float32, copy=False) for x in (tr, va, te))
    t0 = time.time()
    m = gb(params).fit(Xtr, tr[target].values)
    pv, pt = m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]
    res = {"n_features": len(feats), "n_columns": Xtr.shape[1],
           "validation_roc_auc": roc_auc_score(va[target], pv), "validation_pr_auc": average_precision_score(va[target], pv),
           "test_roc_auc": roc_auc_score(te[target], pt), "test_pr_auc": average_precision_score(te[target], pt),
           "fit_seconds": round(time.time() - t0, 1)}
    return res, (prep, feats, m, Xva)


def fields_of(feats, reg) -> list[str]:
    """Raw table.field entries a feature set needs (row counts appear as table.n_rows)."""
    return sorted({f"{reg.loc[f, 'table']}.{reg.loc[f, 'field']}" for f in feats})


def field_funnel(reg: pd.DataFrame) -> pd.DataFrame:
    """Account for every raw field, using the roles recorded in the data dictionary."""
    d = pd.read_excel(PROJECT_ROOT / "documentation" / "Data_Dictionary.xlsx", sheet_name="CRSS fields")
    role = d.project_role.astype(str)
    used = d[role.str.startswith("USED")]
    vin_used = used.table.str.startswith("vpic") | used.field.str.upper().str.startswith("VPIC")
    post_fields = set(zip(reg[reg.group == "post_crash"].table, reg[reg.group == "post_crash"].field))
    post_used = [(t, f) in post_fields for t, f in zip(used.table, used.field)]
    excluded = d[role.str.startswith("EXCLUDED")]
    rows = [
        ("Raw fields in the 28 CRSS tables (counted per table)", len(d), ""),
        ("Excluded by rule: leakage, keys, identifiers, sampling design, VIN text labels",
         -len(excluded), f"{excluded.field.nunique()} distinct field names"),
        ("Removed as duplicate information: text labels, repeats, replaced by imputed versions, constant",
         -(len(d) - len(excluded) - len(used)), ""),
        ("Fields used to build the crash features (benchmark models)", len(used), "summarised into 1,456 crash features"),
        ("Removed after the ablation test: VIN-decoded specification fields (no predictive gain)", -int(vin_used.sum()), ""),
        ("Removed: post-crash fields (recorded after the crash; mostly reflect the outcome)",
         -int(sum(p and not v for p, v in zip(post_used, vin_used))), ""),
        ("Fields available to importance-based selection", int(len(used) - vin_used.sum() - sum(p and not v for p, v in zip(post_used, vin_used))), ""),
    ]
    return pd.DataFrame(rows, columns=["step", "fields", "note"])


def main() -> None:
    reg = pd.read_csv(TABLE_DIR / "crss_feature_register.csv").set_index("feature")
    vin, post = is_vin(reg), reg.group == "post_crash"
    funnel = field_funnel(reg)
    funnel.to_csv(OUT / "crss_field_funnel.csv", index=False)
    print(funnel.to_string(index=False), flush=True)

    df = train_crss.load_features()
    ablation, selected = [], {}
    for target in ["y_injury", "y_serious"]:
        known = df[target] >= 0
        cols = reg.index.tolist() + [target, "YEAR"]
        parts = [df.loc[known & m, cols] for m in (df.YEAR.isin(CRSS_TRAIN_YEARS), df.YEAR == CRSS_VALIDATION_YEAR,
                                                    df.YEAR == CRSS_TEST_YEAR)]
        params = benchmark_params(target)

        # 1-2. Ablation: the benchmark (from train_crss.py), then without VIN, then also without post-crash.
        c1 = pd.read_csv(EVALUATION_DIR / f"crss_{target}_model_comparison.csv")
        c1 = c1[c1.selected].iloc[0]
        ablation.append({"target": target, "step": "Benchmark: all fields (all 28 tables)", "n_features": int(c1.n_input_fields),
                         "n_columns": int(c1.n_encoded_columns), "validation_roc_auc": c1.validation_roc_auc,
                         "validation_pr_auc": c1.validation_pr_auc, "test_roc_auc": c1.test_roc_auc, "test_pr_auc": c1.test_pr_auc})
        r, _ = fit_score(reg.index[~vin].tolist(), parts, reg, target, params)
        ablation.append({"target": target, "step": "Without VIN-decoded fields", **r})
        print(target, "no VIN", {k: round(v, 4) for k, v in r.items()}, flush=True)
        base, (prep, feats, m, Xva) = fit_score(reg.index[~vin & ~post].tolist(), parts, reg, target, params)
        ablation.append({"target": target, "step": "Without VIN-decoded and post-crash fields (selection benchmark)", **base})
        print(target, "benchmark", {k: round(v, 4) for k, v in base.items()}, flush=True)

        # 3. Rank the benchmark's features by permutation importance on validation (same protocol as the benchmark).
        yva = parts[1][target].values
        sample = np.random.default_rng(RNG).choice(len(yva), min(6000, len(yva)), replace=False)
        pi = permutation_importance(m, Xva[sample], yva[sample], scoring="roc_auc", n_repeats=2, random_state=RNG, n_jobs=4)
        names = prep.get_feature_names_out()
        owner = {c: max((f for f in feats if c == f or c.startswith(f + "_")), key=len) for c in names}
        imp = (pd.DataFrame({"feature": [owner[c] for c in names], "importance": pi.importances_mean})
               .groupby("feature", as_index=False)["importance"].sum().sort_values("importance", ascending=False))
        imp["table"], imp["field"] = imp.feature.map(reg.table), imp.feature.map(reg.field)
        imp.to_csv(OUT / f"crss_{target}_selection_importance.csv", index=False)
        ranked = imp.feature.tolist()

        # Retrain on the top-k features and apply the 98% rule on validation.
        rows = []
        for k in KS:
            top = ranked[:k]
            r, _ = fit_score(top, parts, reg, target, params)
            r.update({"k": k, "n_fields": len(fields_of(top, reg)), "n_tables": len({reg.loc[f, "table"] for f in top}),
                      "share_of_benchmark_validation_pr_auc": r["validation_pr_auc"] / base["validation_pr_auc"],
                      "share_of_benchmark_test_pr_auc": r["test_pr_auc"] / base["test_pr_auc"]})
            rows.append(r)
            print(f"  {target} k={k:3d} fields={r['n_fields']:3d} val PR-AUC {r['validation_pr_auc']:.4f} "
                  f"({r['share_of_benchmark_validation_pr_auc']:.1%})", flush=True)
        topk = pd.DataFrame(rows)
        topk["meets_rule"] = topk.share_of_benchmark_validation_pr_auc >= RULE
        topk.to_csv(OUT / f"crss_{target}_selection_topk.csv", index=False)
        k = int(topk[topk.meets_rule].k.min())
        chosen = ranked[:k]
        selected[target] = {"rule": f"smallest k with validation PR-AUC >= {RULE:.0%} of the benchmark without VIN-decoded and post-crash fields",
                            "k": k, "features": chosen, "fields": fields_of(chosen, reg),
                            "benchmark_validation_pr_auc": base["validation_pr_auc"]}
        print(f"{target}: selected k={k}, {len(selected[target]['fields'])} fields", flush=True)

    pd.DataFrame(ablation).to_csv(OUT / "crss_selection_ablation.csv", index=False)
    union = sorted(set(selected["y_injury"]["fields"]) | set(selected["y_serious"]["fields"]))
    selected["deployment_fields"] = union
    (OUT / "crss_selected_features.json").write_text(json.dumps(selected, indent=1))
    print(f"Fields the deployment must supply (union of S1 and S2): {len(union)}")


if __name__ == "__main__":
    main()
