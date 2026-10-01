"""Train and compare text models on complaint narratives.

U1  component routing   - multi-label (19 harmonised component groups)
U2  serious-incident flag - binary (complaint reports crash / fire / injury / death)

Split: train 2010-2021 (narratives before 2010 use an older component taxonomy and style),
validation 2022-2023 (tuning + thresholds), test 2024-2026 (scored once).
TF-IDF vocabulary is fitted on training narratives only.
Outputs: outputs/evaluation/text_*.csv|png, models/text_<task>.joblib
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
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from project_paths import EVALUATION_DIR, PROCESSED_DIR, PROJECT_ROOT

MODEL_DIR = PROJECT_ROOT / "models"
TRAIN_FROM = 2010
RNG = 42


def scores(model, X):
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        return p[:, 1] if p.ndim == 2 and p.shape[1] == 2 and not isinstance(model, OneVsRestClassifier) else p
    return model.decision_function(X)


def thresholds(Y, S) -> np.ndarray:
    Y, S = np.atleast_2d(Y.T).T, np.atleast_2d(S.T).T
    out = []
    for j in range(Y.shape[1]):
        prec, rec, thr = precision_recall_curve(Y[:, j], S[:, j])
        f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
        out.append(thr[np.nanargmax(f1[:-1])] if len(thr) else 0.5)
    return np.array(out)


def binary_metrics(y, s, thr) -> dict:
    pred = (s >= thr).astype(int)
    return {"roc_auc": roc_auc_score(y, s), "pr_auc": average_precision_score(y, s),
            "precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred), "f1": f1_score(y, pred)}


def multilabel_metrics(Y, S, thr) -> dict:
    P = (S >= thr).astype(int)
    # Guarantee one recommendation per complaint: the top-scoring group if nothing passes its threshold.
    S_rel = S - thr
    none = P.sum(1) == 0
    P[none, S_rel[none].argmax(1)] = 1
    top1 = S_rel.argmax(1)
    return {"micro_f1": f1_score(Y, P, average="micro", zero_division=0),
            "macro_f1": f1_score(Y, P, average="macro", zero_division=0),
            "weighted_f1": f1_score(Y, P, average="weighted", zero_division=0),
            "top1_accuracy": float(Y[np.arange(len(Y)), top1].mean()),
            "macro_pr_auc": average_precision_score(Y, S, average="macro"),
            "exact_match": float((P == Y).all(1).mean())}


def candidates(task: str) -> dict:
    wrap = (lambda m: OneVsRestClassifier(m, n_jobs=4)) if task == "U1" else (lambda m: m)
    return {
        "Dummy (prior)": [wrap(DummyClassifier(strategy="prior"))],
        "Complement Naive Bayes": [wrap(ComplementNB(alpha=a)) for a in (0.1, 0.5)],
        "Logistic Regression": [wrap(LogisticRegression(C=c, solver="liblinear", max_iter=1000)) for c in (1.0, 4.0)],
        "Linear SVM": [wrap(LinearSVC(C=c)) for c in (0.1, 0.5)],
    }


def main() -> None:
    cols = None
    df = pd.read_parquet(PROCESSED_DIR / "complaints_model.parquet")
    df = df[df.year >= TRAIN_FROM]
    groups = [c for c in df.columns if c.startswith("c__")]
    tr, va, te = (df[df.split == s] for s in ("train", "validation", "test"))
    print({s: len(d) for s, d in zip(("train", "validation", "test"), (tr, va, te))}, flush=True)

    t0 = time.time()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.5, max_features=300000, sublinear_tf=True,
                          lowercase=True, strip_accents="unicode", token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9]+\b",
                          dtype=np.float32)
    Xtr = vec.fit_transform(tr.text); Xva = vec.transform(va.text); Xte = vec.transform(te.text)
    print(f"TF-IDF {Xtr.shape} in {time.time()-t0:.0f}s", flush=True)
    sub = np.random.default_rng(RNG).choice(Xtr.shape[0], 150000, replace=False)  # train-fit diagnostics

    for task in ["U2", "U1"]:
        if task == "U1":
            keep_tr, keep_va, keep_te = (d[groups].sum(1).values > 0 for d in (tr, va, te))
            Ytr, Yva, Yte = tr[groups].values[keep_tr], va[groups].values[keep_va], te[groups].values[keep_te]
            Xtr_, Xva_, Xte_ = Xtr[keep_tr], Xva[keep_va], Xte[keep_te]
        else:
            Ytr, Yva, Yte = tr.y_serious.values, va.y_serious.values, te.y_serious.values
            Xtr_, Xva_, Xte_ = Xtr, Xva, Xte
        sub_ = sub[sub < Xtr_.shape[0]]
        rows, fitted = [], {}
        for family, models in candidates(task).items():
            best = None
            for m in models:
                t1 = time.time()
                m.fit(Xtr_, Ytr)
                S = scores(m, Xva_)
                if task == "U2":
                    val = average_precision_score(Yva, S) if family != "Dummy (prior)" else Yva.mean()
                else:
                    val = average_precision_score(Yva, S, average="macro") if family != "Dummy (prior)" else 0
                print(f"  {task} {family:24s} val PR-AUC={val:.4f} ({time.time()-t1:.0f}s)", flush=True)
                if best is None or val > best[0]:
                    best = (val, m, S, time.time() - t1)
            _, m, Sva, secs = best
            row = {"task": task, "model": family, "params": json.dumps({k: v for k, v in m.get_params().items()
                                                                       if k.split("__")[-1] in ("C", "alpha")}),
                   "fit_seconds": round(secs, 1)}
            if family == "Dummy (prior)":
                rows.append({**row, "validation_pr_auc": float(Yva.mean()) if task == "U2" else float(Yva.mean(0).mean())})
                continue
            thr = thresholds(Yva, Sva)
            thr = thr[0] if task == "U2" else thr
            Ste, Str = scores(m, Xte_), scores(m, Xtr_[sub_])
            f = binary_metrics if task == "U2" else multilabel_metrics
            for name, (Y, S) in {"train": (Ytr[sub_], Str), "validation": (Yva, Sva), "test": (Yte, Ste)}.items():
                row.update({f"{name}_{k}": v for k, v in f(Y, S, thr).items()})
            key = "pr_auc" if task == "U2" else "macro_f1"
            row["overfit_gap_train_minus_val"] = row[f"train_{key}"] - row[f"validation_{key}"]
            rows.append(row)
            fitted[family] = (m, thr, Ste)
        res = pd.DataFrame(rows)
        key = "validation_pr_auc" if task == "U2" else "validation_macro_f1"
        sel = res.dropna(subset=[key]).sort_values(key, ascending=False).iloc[0]
        res["selected"] = res.model == sel.model
        res.to_csv(EVALUATION_DIR / f"text_{task}_model_comparison.csv", index=False)
        print(res.drop(columns=["params"]).round(4).to_string(index=False), flush=True)
        m, thr, Ste = fitted[sel.model]

        # Per-class results (U1) and explanation terms for the selected model.
        terms = np.array(vec.get_feature_names_out())
        est = m.estimators_ if task == "U1" else [m]
        names = [g[3:] for g in groups] if task == "U1" else ["serious incident"]
        expl = {}
        for name, e in zip(names, est):
            w = e.coef_.ravel() if hasattr(e, "coef_") else (e.feature_log_prob_[1] - e.feature_log_prob_[0])
            expl[name] = terms[np.argsort(-w)[:25]].tolist()
        json.dump(expl, open(EVALUATION_DIR / f"text_{task}_top_terms.json", "w"), indent=1)
        if task == "U1":
            P = (Ste >= thr).astype(int)
            per = pd.DataFrame({"component": names, "test_support": Yte.sum(0),
                                "precision": precision_score(Yte, P, average=None, zero_division=0),
                                "recall": recall_score(Yte, P, average=None, zero_division=0),
                                "f1": f1_score(Yte, P, average=None, zero_division=0),
                                "threshold": thr})
            per.to_csv(EVALUATION_DIR / "text_U1_per_component_test.csv", index=False)
            print(per.round(3).to_string(index=False), flush=True)

        # Learning curve for the selected model (overfitting / data-sufficiency check).
        lc = []
        order = np.random.default_rng(RNG).permutation(Xtr_.shape[0])
        for frac in (0.02, 0.1, 0.3, 1.0):
            idx = order[: int(len(order) * frac)]
            m2 = clone(m).fit(Xtr_[idx], Ytr[idx])
            ev = idx[:50000]
            if task == "U2":
                lc.append({"train_rows": len(idx), "train": average_precision_score(Ytr[ev], scores(m2, Xtr_[ev])),
                           "validation": average_precision_score(Yva, scores(m2, Xva_))})
            else:
                lc.append({"train_rows": len(idx),
                           "train": average_precision_score(Ytr[ev], scores(m2, Xtr_[ev]), average="macro"),
                           "validation": average_precision_score(Yva, scores(m2, Xva_), average="macro")})
            print(f"  learning curve {task} {lc[-1]}", flush=True)
        lc = pd.DataFrame(lc); lc.to_csv(EVALUATION_DIR / f"text_{task}_learning_curve.csv", index=False)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(lc.train_rows, lc.train, marker="o", label="train"); ax.plot(lc.train_rows, lc.validation, marker="o", label="validation")
        ax.set_xscale("log"); ax.set_xlabel("training complaints"); ax.set_ylabel("PR-AUC" + (" (macro)" if task == "U1" else ""))
        ax.set_title(f"Learning curve {task}: {sel.model}"); ax.legend(); fig.tight_layout()
        fig.savefig(EVALUATION_DIR / f"text_{task}_learning_curve.png", dpi=150); plt.close(fig)

        MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump({"vectorizer": vec, "model": m, "threshold": thr, "labels": names, "task": task,
                     "model_name": sel.model, "test_metrics": {k: float(sel[k]) for k in sel.index if k.startswith("test_")}},
                    MODEL_DIR / f"text_{task}.joblib", compress=3)


if __name__ == "__main__":
    main()
