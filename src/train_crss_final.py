"""Modelling and Evaluation: the final crash-severity models (S1 injury, S2 serious/fatal).

Trains Gradient Boosting on exactly the features chosen by select_crash_features.py (98% rule), using the
same chronological split as the benchmark: tuned and thresholded on 2023, scored once on 2024. The benchmark models
(train_crss.py, all 1,456 features) are what these final models are compared with.

Gradient Boosting was selected in the benchmark comparison against Logistic Regression, a Decision Tree and Random Forest;
here only its tree size is re-tuned for the smaller feature set. A dummy baseline is reported for reference.

Outputs:
  models/crss_<target>_final.joblib                         the deployable pipeline (preprocessing + model)
  outputs/evaluation/final/crss_<target>_final_results.csv  train / validation / test metrics
  outputs/evaluation/final/crss_<target>_final_*            confusion matrix, calibration, learning curve,
                                                            permutation importance, diagnostics chart
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_crss
from project_paths import (CRSS_TEST_YEAR, CRSS_TRAIN_YEARS, CRSS_VALIDATION_YEAR, EVALUATION_DIR, OUTPUT_DIR,
                           PROJECT_ROOT, TABLE_DIR)
from select_crash_features import gb

OUT = EVALUATION_DIR / "final"
OUT.mkdir(parents=True, exist_ok=True)
MODEL_DIR = PROJECT_ROOT / "models"
RNG = 42


def run(target: str, df: pd.DataFrame, reg: pd.DataFrame, chosen: dict) -> pd.DataFrame:
    feats = chosen["features"]
    known = df[target] >= 0
    tr, va, te = (df.loc[known & m, feats + [target, "WEIGHT"]] for m in
                  (df.YEAR.isin(CRSS_TRAIN_YEARS), df.YEAR == CRSS_VALIDATION_YEAR, df.YEAR == CRSS_TEST_YEAR))
    cat = [f for f in feats if reg.loc[f, "kind"] == "categorical_code"]
    prep = train_crss.preprocessor(cat, [f for f in feats if f not in cat]).fit(tr[feats])
    Xtr, Xva, Xte = (prep.transform(x[feats]).astype(np.float32, copy=False) for x in (tr, va, te))
    ytr, yva, yte = tr[target].values, va[target].values, te[target].values

    # Re-tune tree size on validation only.
    best = None
    for leaves in (31, 63):
        m = gb({"max_leaf_nodes": leaves, "min_samples_leaf": 20}).fit(Xtr, ytr)
        pv = m.predict_proba(Xva)[:, 1]
        auc = roc_auc_score(yva, pv)
        print(f"  {target} leaves={leaves} validation ROC-AUC {auc:.4f}", flush=True)
        if best is None or auc > best[0]:
            best = (auc, leaves, m, pv)
    _, leaves, m, pv = best
    thr = train_crss.best_threshold(yva, pv)
    ptr, pte = m.predict_proba(Xtr)[:, 1], m.predict_proba(Xte)[:, 1]

    rows = []
    for name, model_name in [("dummy", "Dummy (prior)"), ("final", "Gradient Boosting (final)")]:
        row = {"target": target, "model": model_name, "n_features": len(feats), "n_columns": Xtr.shape[1],
               "n_fields": len(chosen["fields"]), "max_leaf_nodes": leaves if name == "final" else None,
               "threshold": thr if name == "final" else None}
        for split, (y, p, w) in {"train": (ytr, ptr, None), "validation": (yva, pv, None),
                                 "test": (yte, pte, te["WEIGHT"].values)}.items():
            if name == "dummy":
                row.update({f"{split}_roc_auc": 0.5, f"{split}_pr_auc": float(y.mean())})
            else:
                row.update({f"{split}_{k}": v for k, v in train_crss.metrics(y, p, thr, w).items()})
        if name == "final":
            row["overfit_gap_auc_train_minus_val"] = row["train_roc_auc"] - row["validation_roc_auc"]
            row["drift_gap_auc_val_minus_test"] = row["validation_roc_auc"] - row["test_roc_auc"]
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / f"crss_{target}_final_results.csv", index=False)

    cm = confusion_matrix(yte, (pte >= thr).astype(int))
    pd.DataFrame(cm, index=["actual 0", "actual 1"], columns=["pred 0", "pred 1"]).to_csv(OUT / f"crss_{target}_final_confusion_test.csv")
    frac, mean = calibration_curve(yte, pte, n_bins=10)
    pd.DataFrame({"mean_predicted": mean, "observed_rate": frac}).to_csv(OUT / f"crss_{target}_final_calibration_test.csv", index=False)

    lc = []
    order = np.random.default_rng(RNG).permutation(len(ytr))
    for frac_ in (0.05, 0.1, 0.25, 0.5, 1.0):
        idx = order[: int(len(order) * frac_)]
        m2 = clone(m).fit(Xtr[idx], ytr[idx])
        lc.append({"train_rows": len(idx), "train_auc": roc_auc_score(ytr[idx], m2.predict_proba(Xtr[idx])[:, 1]),
                   "validation_auc": roc_auc_score(yva, m2.predict_proba(Xva)[:, 1])})
    lc = pd.DataFrame(lc)
    lc.to_csv(OUT / f"crss_{target}_final_learning_curve.csv", index=False)

    cols = prep.get_feature_names_out()
    owner = {c: max((f for f in feats if c == f or c.startswith(f + "_")), key=len) for c in cols}
    sample = np.random.default_rng(RNG).choice(len(yva), min(6000, len(yva)), replace=False)
    pi = permutation_importance(m, Xva[sample], yva[sample], scoring="roc_auc", n_repeats=2, random_state=RNG, n_jobs=4)
    imp = (pd.DataFrame({"feature": [owner[c] for c in cols], "importance": pi.importances_mean})
           .groupby("feature", as_index=False)["importance"].sum().sort_values("importance", ascending=False))
    imp["table"], imp["field"] = imp.feature.map(reg.table), imp.feature.map(reg.field)
    imp.to_csv(OUT / f"crss_{target}_final_permutation_importance.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(lc.train_rows, lc.train_auc, marker="o", label="train")
    ax[0].plot(lc.train_rows, lc.validation_auc, marker="o", label="validation (2023)")
    ax[0].set_xscale("log"); ax[0].set_xlabel("training crashes"); ax[0].set_ylabel("ROC-AUC")
    ax[0].set_title(f"Learning curve: final model ({len(feats)} features)"); ax[0].legend()
    ax[1].plot([0, 1], [0, 1], ls="--", color="grey"); ax[1].plot(mean, frac, marker="o")
    ax[1].set_xlabel("predicted probability"); ax[1].set_ylabel("observed rate (2024)"); ax[1].set_title("Calibration (test)")
    fig.tight_layout(); fig.savefig(OUT / f"crss_{target}_final_diagnostics.png", dpi=150); plt.close(fig)

    final = res[res.model != "Dummy (prior)"].iloc[0]
    joblib.dump({"pipeline": Pipeline([("prep", prep), ("model", m)]), "features": feats, "fields": chosen["fields"],
                 "threshold": thr, "target": target, "model": "Gradient Boosting (final)",
                 "selection_rule": chosen["rule"],
                 "test_metrics": {k: float(final[k]) for k in final.index if k.startswith("test_")},
                 "top_features": imp.head(25).to_dict("records")},
                MODEL_DIR / f"crss_{target}_final.joblib", compress=3)
    print(res.drop(columns=[c for c in res.columns if c.startswith("train_")]).round(4).to_string(index=False), flush=True)
    return res


def main() -> None:
    reg = pd.read_csv(TABLE_DIR / "crss_feature_register.csv").set_index("feature")
    selected = json.loads((OUTPUT_DIR / "selection" / "crss_selected_features.json").read_text())
    df = train_crss.load_features()
    for target in sys.argv[1:] or ["y_injury", "y_serious"]:
        run(target, df, reg, selected[target])


if __name__ == "__main__":
    main()
