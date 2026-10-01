"""Train and compare structured crash-severity models (S1 injury, S2 serious/fatal).

Split: train 2020-2022, validation 2023 (all tuning and threshold choice), test 2024 (scored once).
Feature sets: 'all' fields vs 'no_post_crash' (excludes damage/towing/test-result fields) to show
the models do not depend on information recorded after the crash.
Outputs: outputs/evaluation/crss_*.csv|png, models/crss_<target>.joblib
"""
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from project_paths import (CRSS_TEST_YEAR, CRSS_TRAIN_YEARS, CRSS_VALIDATION_YEAR, EVALUATION_DIR,
                           PROCESSED_DIR, PROJECT_ROOT, TABLE_DIR)

MODEL_DIR = PROJECT_ROOT / "models"
RNG = 42


def candidates(balanced: bool) -> dict:
    cw = "balanced" if balanced else None
    lr = lambda c: Pipeline([("scale", StandardScaler()), ("lr", LogisticRegression(C=c, max_iter=3000, class_weight=cw))])
    return {
        "Dummy (prior)": [DummyClassifier(strategy="prior")],
        "Logistic Regression": [lr(c) for c in (0.01, 0.1, 1.0)],
        "Decision Tree": [DecisionTreeClassifier(max_depth=d, min_samples_leaf=50, class_weight=cw, random_state=RNG)
                          for d in (6, 10, 14)],
        "Random Forest": [RandomForestClassifier(n_estimators=300, min_samples_leaf=l, max_features="sqrt",
                                                 class_weight=cw, n_jobs=-1, random_state=RNG) for l in (5, 20)],
        "Gradient Boosting": [HistGradientBoostingClassifier(learning_rate=0.08, max_leaf_nodes=n, max_iter=600,
                                                             l2_regularization=1.0, early_stopping=True,
                                                             validation_fraction=0.1, n_iter_no_change=30,
                                                             class_weight=cw, random_state=RNG) for n in (31, 63)],
    }


def describe_params(m) -> str:
    return json.dumps({k.split("__")[-1]: v for k, v in m.get_params().items()
                       if k.split("__")[-1] in ("C", "max_depth", "min_samples_leaf", "max_leaf_nodes")})


def preprocessor(cat: list[str], num: list[str]) -> ColumnTransformer:
    """Fitted once per feature set on training years; scaling happens inside the LR pipeline."""
    return ColumnTransformer([
        ("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value=-9)),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32))]), cat),
        ("num", SimpleImputer(strategy="median"), num),
    ], verbose_feature_names_out=False)


def metrics(y, p, thr, w=None) -> dict:
    pred = (p >= thr).astype(int)
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred),
            "f1": f1_score(y, pred), "brier": brier_score_loss(y, p),
            **({"weighted_roc_auc": roc_auc_score(y, p, sample_weight=w)} if w is not None else {})}


def best_threshold(y, p) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    return float(thr[np.nanargmax(f1[:-1])])


def drop_redundant(X: pd.DataFrame) -> list[str]:
    """Remove constant and exactly duplicated columns (decided on training data only)."""
    X = X.loc[:, X.nunique(dropna=False) > 1]
    return X.T.drop_duplicates().index.tolist()


def run(target: str, df: pd.DataFrame, reg: pd.DataFrame) -> pd.DataFrame:
    d = df[df[target] >= 0]
    tr, va, te = (d[d.YEAR.isin(CRSS_TRAIN_YEARS)], d[d.YEAR == CRSS_VALIDATION_YEAR], d[d.YEAR == CRSS_TEST_YEAR])
    rows, fitted, arrays = [], {}, {}
    for fset in ["all", "no_post_crash"]:
        feats = reg.index.tolist() if fset == "all" else reg.index[reg.group != "post_crash"].tolist()
        feats = drop_redundant(tr[feats])
        cat = [f for f in feats if reg.loc[f, "kind"] == "categorical_code"]
        num = [f for f in feats if f not in cat]
        prep = preprocessor(cat, num).fit(tr[feats])
        Xtr, Xva, Xte = (prep.transform(x[feats]).astype(np.float32) for x in (tr, va, te))
        arrays[fset] = (prep, feats, Xtr, Xva, Xte)
        balanced_options = [False, True] if target == "y_serious" else [False]
        for balanced in balanced_options:
            for family, models in candidates(balanced).items():
                if family == "Dummy (prior)" and balanced:
                    continue
                best = None
                for m in models:
                    t0 = time.time()
                    m.fit(Xtr, tr[target].values)
                    pv = m.predict_proba(Xva)[:, 1]
                    auc = roc_auc_score(va[target], pv) if family != "Dummy (prior)" else 0.5
                    print(f"  {target} {fset:13s} bal={balanced!s:5s} {family:20s} {describe_params(m):34s} "
                          f"val AUC={auc:.4f} ({time.time()-t0:.0f}s)", flush=True)
                    if best is None or auc > best[0]:
                        best = (auc, m, pv, time.time() - t0)
                _, m, pv, secs = best
                thr = best_threshold(va[target], pv) if family != "Dummy (prior)" else 0.5
                ptr = m.predict_proba(Xtr)[:, 1]
                pte = m.predict_proba(Xte)[:, 1]
                row = {"target": target, "feature_set": fset, "class_weight": "balanced" if balanced else "none",
                       "model": family, "params": describe_params(m), "n_input_fields": len(feats),
                       "n_encoded_columns": Xtr.shape[1], "fit_seconds": round(secs, 1), "threshold": thr}
                for name, (y, p, w) in {"train": (tr[target], ptr, None), "validation": (va[target], pv, None),
                                         "test": (te[target], pte, te["WEIGHT"])}.items():
                    if family == "Dummy (prior)":
                        row.update({f"{name}_roc_auc": 0.5, f"{name}_pr_auc": float(y.mean())})
                        continue
                    for k, v in metrics(y, p, thr, w).items():
                        row[f"{name}_{k}"] = v
                if family != "Dummy (prior)":
                    row["overfit_gap_auc_train_minus_val"] = row["train_roc_auc"] - row["validation_roc_auc"]
                    row["drift_gap_auc_val_minus_test"] = row["validation_roc_auc"] - row["test_roc_auc"]
                rows.append(row)
                fitted[(fset, balanced, family)] = (m, thr, pte)
    res = pd.DataFrame(rows)

    # Select on validation only: best PR-AUC among non-dummy models with the full feature set.
    cand = res[(res.model != "Dummy (prior)")].sort_values("validation_pr_auc", ascending=False)
    sel = cand.iloc[0]
    key = (sel.feature_set, sel.class_weight == "balanced", sel.model)
    m, thr, pte = fitted[key]
    prep, feats, Xtr, Xva, Xte = arrays[sel.feature_set]
    pipe = Pipeline([("prep", prep), ("model", m)])
    res["selected"] = (res.feature_set == sel.feature_set) & (res.class_weight == sel.class_weight) & (res.model == sel.model)
    ytr, yva = tr[target].values, va[target].values

    cm = confusion_matrix(te[target], (pte >= thr).astype(int))
    pd.DataFrame(cm, index=["actual 0", "actual 1"], columns=["pred 0", "pred 1"]).to_csv(EVALUATION_DIR / f"crss_{target}_confusion_test.csv")
    frac, mean = calibration_curve(te[target], pte, n_bins=10)
    pd.DataFrame({"mean_predicted": mean, "observed_rate": frac}).to_csv(EVALUATION_DIR / f"crss_{target}_calibration_test.csv", index=False)

    # Learning curve (overfitting diagnosis): refit on growing fractions of the training years.
    lc = []
    order = np.random.default_rng(RNG).permutation(len(ytr))
    for frac_ in (0.05, 0.1, 0.25, 0.5, 1.0):
        idx = order[: int(len(order) * frac_)]
        m2 = clone(m).fit(Xtr[idx], ytr[idx])
        lc.append({"train_rows": len(idx), "train_auc": roc_auc_score(ytr[idx], m2.predict_proba(Xtr[idx])[:, 1]),
                   "validation_auc": roc_auc_score(yva, m2.predict_proba(Xva)[:, 1])})
        print(f"  learning curve {lc[-1]}", flush=True)
    lc = pd.DataFrame(lc)
    lc.to_csv(EVALUATION_DIR / f"crss_{target}_learning_curve.csv", index=False)

    # Permutation importance on validation rows, aggregated back to the original field.
    cols = prep.get_feature_names_out()
    owner = {c: max((f for f in feats if c == f or c.startswith(f + "_")), key=len) for c in cols}
    sample = np.random.default_rng(RNG).choice(len(yva), min(6000, len(yva)), replace=False)
    pi = permutation_importance(m, Xva[sample], yva[sample], scoring="roc_auc", n_repeats=2, random_state=RNG, n_jobs=4)
    imp = (pd.DataFrame({"column": cols, "feature": [owner[c] for c in cols], "importance": pi.importances_mean})
           .groupby("feature", as_index=False)["importance"].sum().sort_values("importance", ascending=False))
    imp["table"] = imp.feature.map(reg["table"]); imp["group"] = imp.feature.map(reg["group"])
    imp.to_csv(EVALUATION_DIR / f"crss_{target}_permutation_importance.csv", index=False)
    print(imp.head(20).to_string(index=False), flush=True)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(lc.train_rows, lc.train_auc, marker="o", label="train")
    ax[0].plot(lc.train_rows, lc.validation_auc, marker="o", label="validation (2023)")
    ax[0].set_xscale("log"); ax[0].set_xlabel("training crashes"); ax[0].set_ylabel("ROC-AUC")
    ax[0].set_title(f"Learning curve: {sel.model}"); ax[0].legend()
    ax[1].plot([0, 1], [0, 1], ls="--", color="grey"); ax[1].plot(mean, frac, marker="o")
    ax[1].set_xlabel("predicted probability"); ax[1].set_ylabel("observed rate (2024)"); ax[1].set_title("Calibration (test)")
    fig.tight_layout(); fig.savefig(EVALUATION_DIR / f"crss_{target}_diagnostics.png", dpi=150); plt.close(fig)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"pipeline": pipe, "features": feats, "threshold": thr, "target": target,
                 "model": sel.model, "feature_set": sel.feature_set, "class_weight": sel.class_weight,
                 "test_metrics": {k: float(sel[k]) for k in sel.index if k.startswith("test_")},
                 "top_features": imp.head(25).to_dict("records")},
                MODEL_DIR / f"crss_{target}.joblib", compress=3)
    return res


def main() -> None:
    df = pd.read_parquet(PROCESSED_DIR / "crss_crash_features.parquet")
    reg = pd.read_csv(TABLE_DIR / "crss_feature_register.csv").set_index("feature")
    targets = sys.argv[1:] or ["y_injury", "y_serious"]
    for t in targets:
        res = run(t, df, reg)
        res.to_csv(EVALUATION_DIR / f"crss_{t}_model_comparison.csv", index=False)
        cols = ["feature_set", "class_weight", "model", "validation_roc_auc", "validation_pr_auc", "test_roc_auc",
                "test_pr_auc", "test_f1", "overfit_gap_auc_train_minus_val", "selected"]
        print(res[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
