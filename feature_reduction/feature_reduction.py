"""Feature reduction study for the crash-severity models (S1 injury, S2 serious/fatal).

Question: how many of the ~1,300 crash features are needed, and which reduction technique
keeps the most performance with the fewest features?

Techniques compared (all fitted on training years 2020-2022 only):
  Filter    - near-constant removal, correlation pruning, mutual-information top-k
  Embedded  - L1 (Lasso) logistic regression; permutation-importance top-k
  Reduction - PCA components
Each reduced feature set is scored with the same Gradient Boosting configuration as the selected
research model; PCA and L1 are also scored with Logistic Regression. Selection uses validation
(2023); the test year (2024) is reported for reference only.

Outputs: feature_reduction/results/*.csv, *.png
"""
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import train_crss
from project_paths import CRSS_TEST_YEAR, CRSS_TRAIN_YEARS, CRSS_VALIDATION_YEAR, EVALUATION_DIR, TABLE_DIR

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
KS = [10, 25, 50, 100, 200, 400]
RNG = 42


def gb(params: dict) -> HistGradientBoostingClassifier:
    """Same configuration as the selected research model (notebook 04)."""
    return HistGradientBoostingClassifier(learning_rate=0.08, max_iter=600, l2_regularization=1.0,
                                          early_stopping=True, validation_fraction=0.1, n_iter_no_change=30,
                                          random_state=RNG, **params)


def score(model, Xtr, ytr, Xva, yva, Xte, yte) -> dict:
    t0 = time.time()
    model.fit(Xtr, ytr)
    pv, pt = model.predict_proba(Xva)[:, 1], model.predict_proba(Xte)[:, 1]
    return {"validation_roc_auc": roc_auc_score(yva, pv), "validation_pr_auc": average_precision_score(yva, pv),
            "test_roc_auc": roc_auc_score(yte, pt), "test_pr_auc": average_precision_score(yte, pt),
            "fit_seconds": round(time.time() - t0, 1)}


def prepare(target: str, reg: pd.DataFrame):
    """Rebuild exactly the matrices the selected research model used (all fields)."""
    df = train_crss.load_features()
    cols = reg.index.tolist() + [target, "YEAR"]
    known = df[target] >= 0
    tr, va, te = (df.loc[known & m, cols] for m in (df.YEAR.isin(CRSS_TRAIN_YEARS), df.YEAR == CRSS_VALIDATION_YEAR,
                                                     df.YEAR == CRSS_TEST_YEAR))
    del df
    feats = train_crss.drop_redundant(tr[reg.index.tolist()])
    cat = [f for f in feats if reg.loc[f, "kind"] == "categorical_code"]
    num = [f for f in feats if f not in cat]
    prep = train_crss.preprocessor(cat, num).fit(tr[feats])
    names = np.asarray(prep.get_feature_names_out())
    X = [prep.transform(x[feats]).astype(np.float32, copy=False) for x in (tr, va, te)]
    y = [x[target].to_numpy() for x in (tr, va, te)]
    owner = {c: max((f for f in feats if c == f or c.startswith(f + "_")), key=len) for c in names}
    return X, y, names, owner


def run(target: str, reg: pd.DataFrame) -> pd.DataFrame:
    (Xtr, Xva, Xte), (ytr, yva, yte), names, owner = prepare(target, reg)
    sel = pd.read_csv(EVALUATION_DIR / f"crss_{target}_model_comparison.csv")
    sel = sel[sel.selected].iloc[0]
    gb_params = {k: v for k, v in json.loads(sel.params).items() if v is not None and k in ("max_leaf_nodes", "min_samples_leaf")}
    rows, notes = [], {}
    p = Xtr.shape[1]
    print(f"{target}: {Xtr.shape[0]:,} training crashes x {p} encoded columns | GB {gb_params}", flush=True)

    def add(method, family, k, model_name, res):
        rows.append({"target": target, "method": method, "family": family, "n_features": k, "model": model_name, **res})
        print(f"  {method:28s} k={k:5d} {model_name:20s} val PR-AUC {res['validation_pr_auc']:.4f} ({res['fit_seconds']}s)", flush=True)

    # Reference: all columns.
    add("All features", "reference", p, "Gradient Boosting", score(gb(gb_params), Xtr, ytr, Xva, yva, Xte, yte))

    # Filter 1: near-constant columns (non-zero in fewer than 0.1% of training crashes).
    nonzero = (Xtr != 0).mean(axis=0)
    keep_var = np.where(nonzero >= 0.001)[0]
    notes["near_constant_removed"] = int(p - len(keep_var))
    add("Near-constant removed", "filter", len(keep_var), "Gradient Boosting",
        score(gb(gb_params), Xtr[:, keep_var], ytr, Xva[:, keep_var], yva, Xte[:, keep_var], yte))

    # Filter 2: correlation pruning (|r| > 0.95 with an earlier-kept column), on a training sample.
    rng = np.random.default_rng(RNG)
    sample = rng.choice(Xtr.shape[0], min(40000, Xtr.shape[0]), replace=False)
    S = Xtr[sample][:, keep_var].astype(np.float64)
    S = (S - S.mean(0)) / np.where(S.std(0) > 0, S.std(0), 1)
    corr = np.abs(S.T @ S / len(S))
    keep_corr = []
    for j in range(corr.shape[0]):
        if not keep_corr or corr[j, keep_corr].max() <= 0.95:
            keep_corr.append(j)
    keep_corr = keep_var[keep_corr]
    notes["correlated_removed_after_near_constant"] = int(len(keep_var) - len(keep_corr))
    del S, corr
    add("Correlation pruned (|r|>0.95)", "filter", len(keep_corr), "Gradient Boosting",
        score(gb(gb_params), Xtr[:, keep_corr], ytr, Xva[:, keep_corr], yva, Xte[:, keep_corr], yte))

    # Filter 3: mutual information ranking (training sample).
    mi = mutual_info_classif(Xtr[sample], ytr[sample], discrete_features=False, n_neighbors=3, random_state=RNG)
    mi_rank = np.argsort(-mi)
    pd.DataFrame({"column": names[mi_rank], "field": [owner[c] for c in names[mi_rank]], "mutual_information": mi[mi_rank]}) \
        .to_csv(OUT / f"{target}_mutual_information_ranking.csv", index=False)
    for k in KS:
        idx = mi_rank[:k]
        add("Mutual information top-k", "filter", k, "Gradient Boosting",
            score(gb(gb_params), Xtr[:, idx], ytr, Xva[:, idx], yva, Xte[:, idx], yte))

    # Embedded 1: permutation-importance ranking from the selected research model (computed on validation in
    # notebook 05 and aggregated per field; here mapped back to encoded columns).
    imp = pd.read_csv(EVALUATION_DIR / f"crss_{target}_permutation_importance.csv").set_index("feature")["importance"]
    col_imp = np.array([imp.get(owner[c], 0.0) for c in names])
    perm_rank = np.argsort(-col_imp)
    for k in KS:
        idx = perm_rank[:k]
        add("Permutation importance top-k", "embedded", k, "Gradient Boosting",
            score(gb(gb_params), Xtr[:, idx], ytr, Xva[:, idx], yva, Xte[:, idx], yte))

    # Embedded 2: L1 (Lasso) logistic regression; smaller C keeps fewer features.
    scaler = StandardScaler().fit(Xtr)
    Ztr, Zva, Zte = (scaler.transform(x).astype(np.float32) for x in (Xtr, Xva, Xte))
    l1_sets = {}
    for C in (0.0002, 0.0005, 0.001, 0.003, 0.01, 0.05):
        lr = LogisticRegression(penalty="l1", C=C, solver="liblinear", max_iter=2000)
        res = score(lr, Ztr, ytr, Zva, yva, Zte, yte)
        kept = np.flatnonzero(np.abs(lr.coef_.ravel()) > 1e-8)
        l1_sets[C] = kept
        add(f"L1 logistic (C={C})", "embedded", len(kept), "Logistic Regression", res)
    best_C = max(l1_sets, key=lambda c: [r for r in rows if r["method"] == f"L1 logistic (C={c})"][0]["validation_pr_auc"])
    kept = l1_sets[best_C]
    if len(kept) > 0:
        add(f"L1-selected (C={best_C}) then GB", "embedded", len(kept), "Gradient Boosting",
            score(gb(gb_params), Xtr[:, kept], ytr, Xva[:, kept], yva, Xte[:, kept], yte))

    # Dimensionality reduction: PCA on standardised columns (fitted on training data).
    pca = PCA(n_components=200, svd_solver="randomized", random_state=RNG).fit(Ztr[sample])
    evr = np.cumsum(pca.explained_variance_ratio_)
    notes["pca_variance_explained"] = {k: round(float(evr[k - 1]), 3) for k in (10, 25, 50, 100, 200)}
    Ptr, Pva, Pte = (pca.transform(z).astype(np.float32) for z in (Ztr, Zva, Zte))
    del Ztr, Zva, Zte
    for k in (10, 25, 50, 100, 200):
        add("PCA components", "dimensionality reduction", k, "Gradient Boosting",
            score(gb(gb_params), Ptr[:, :k], ytr, Pva[:, :k], yva, Pte[:, :k], yte))
        add("PCA components", "dimensionality reduction", k, "Logistic Regression",
            score(LogisticRegression(C=0.1, max_iter=2000), Ptr[:, :k], ytr, Pva[:, :k], yva, Pte[:, :k], yte))

    # Top fields under the two rankings, for interpretation.
    top = pd.DataFrame({"mutual_information_top_fields": list(dict.fromkeys(owner[c] for c in names[mi_rank[:60]]))[:25]})
    top["permutation_importance_top_fields"] = pd.Series(list(dict.fromkeys(owner[c] for c in names[perm_rank[:60]]))[:25])
    top.to_csv(OUT / f"{target}_top_fields.csv", index=False)
    (OUT / f"{target}_notes.json").write_text(json.dumps({**notes, "encoded_columns": int(p),
                                                          "l1_best_C": best_C}, indent=1))
    return pd.DataFrame(rows)


def plot(res: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    styles = {"Mutual information top-k": ("tab:blue", "o"), "Permutation importance top-k": ("tab:green", "s"),
              "PCA components|Gradient Boosting": ("tab:purple", "^"), "PCA components|Logistic Regression": ("tab:purple", "v")}
    for ax, (t, label) in zip(axes, [("y_injury", "S1 injury crash"), ("y_serious", "S2 serious or fatal crash")]):
        r = res[res.target == t]
        full = r[r.method == "All features"].validation_pr_auc.iloc[0]
        for key, (color, marker) in styles.items():
            method, _, model = key.partition("|")
            d = r[(r.method == method) & ((r.model == model) if model else True)].sort_values("n_features")
            ax.plot(d.n_features, d.validation_pr_auc, marker=marker, color=color,
                    ls="--" if model == "Logistic Regression" else "-", label=key.replace("|", " + "))
        l1 = r[r.method.str.startswith("L1 logistic")]
        ax.scatter(l1.n_features, l1.validation_pr_auc, color="tab:orange", marker="D", label="L1 logistic (different C)", zorder=3)
        for m, c in [("Near-constant removed", "tab:gray"), ("Correlation pruned (|r|>0.95)", "tab:brown")]:
            d = r[r.method == m]
            ax.scatter(d.n_features, d.validation_pr_auc, color=c, marker="X", s=60, label=m, zorder=3)
        ax.axhline(full, color="black", lw=1, ls=":", label=f"All {int(r[r.method == 'All features'].n_features.iloc[0])} features")
        ax.set_xscale("log"); ax.set_xlabel("number of features / components (log scale)")
        ax.set_ylabel("validation PR-AUC"); ax.set_title(label)
    axes[1].legend(fontsize=7.5, loc="lower right")
    fig.suptitle("How many crash features are needed? Feature reduction compared (validation 2023)")
    fig.tight_layout(); fig.savefig(OUT / "feature_reduction_comparison.png", dpi=150); plt.close(fig)


def summarise(res: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t in ("y_injury", "y_serious"):
        r = res[res.target == t]
        full = r[r.method == "All features"].iloc[0]
        for method in ["Mutual information top-k", "Permutation importance top-k", "PCA components"]:
            d = r[(r.method == method) & (r.model == "Gradient Boosting")].sort_values("n_features")
            for share in (0.95, 0.99):
                hit = d[d.validation_pr_auc >= share * full.validation_pr_auc]
                rows.append({"target": t, "method": method, "retain": f"{share:.0%} of full validation PR-AUC",
                             "features_needed": int(hit.n_features.iloc[0]) if len(hit) else None,
                             "full_features": int(full.n_features)})
    return pd.DataFrame(rows)


def main() -> None:
    reg = pd.read_csv(TABLE_DIR / "crss_feature_register.csv").set_index("feature")
    targets = sys.argv[1:] or ["y_injury", "y_serious"]
    for t in targets:
        run(t, reg).to_csv(OUT / f"{t}_results.csv", index=False)
    res = pd.concat([pd.read_csv(OUT / f"{t}_results.csv") for t in ("y_injury", "y_serious") if (OUT / f"{t}_results.csv").exists()])
    res.to_csv(OUT / "all_results.csv", index=False)
    if res.target.nunique() == 2:
        plot(res)
        s = summarise(res); s.to_csv(OUT / "features_needed_summary.csv", index=False)
        print(s.to_string(index=False))


if __name__ == "__main__":
    main()
