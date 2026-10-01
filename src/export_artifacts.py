"""Copy deployment artifacts into app/api/artifacts and build the model card.

Run after train_crss.py, train_text.py, train_crash_deploy.py and build_vehicle_profiles.py.
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from project_paths import EVALUATION_DIR, PROCESSED_DIR, PROJECT_ROOT

ART = PROJECT_ROOT / "app" / "api" / "artifacts"


def selected(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    return df[df.selected].iloc[0]


def baseline_pr_auc(path: Path) -> float:
    df = pd.read_csv(path)
    return float(df[df.model == "Dummy (prior)"].iloc[0].test_pr_auc)


def main() -> None:
    ART.mkdir(parents=True, exist_ok=True)
    for task in ("U1", "U2"):
        bundle = joblib.load(PROJECT_ROOT / "models" / f"text_{task}.joblib")
        # stop_words_ only records terms pruned during fitting; it is not used for transform().
        bundle["vectorizer"].stop_words_ = None
        joblib.dump(bundle, ART / f"text_{task}.joblib", compress=3)

    prof = pd.read_parquet(PROCESSED_DIR / "vehicle_profiles.parquet")
    prof.to_parquet(ART / "vehicle_profiles.parquet", index=False)

    s1, s2 = (selected(EVALUATION_DIR / f"crss_{t}_model_comparison.csv") for t in ("y_injury", "y_serious"))
    u1, u2 = (selected(EVALUATION_DIR / f"text_{t}_model_comparison.csv") for t in ("U1", "U2"))
    deploy = pd.read_csv(EVALUATION_DIR / "crash_deploy_comparison.csv")
    dep = {t: deploy[deploy.target == t].sort_values("validation_pr_auc").iloc[-1] for t in ("y_injury", "y_serious")}
    card = {"models": [
        {"task": "S1 Injury crash (CRSS)", "model": s1.model, "headline_metric": "ROC-AUC, 2024 test",
         "headline_value": f"{s1.test_roc_auc:.3f}",
         "detail": f"PR-AUC {s1.test_pr_auc:.3f} vs baseline {baseline_pr_auc(EVALUATION_DIR / 'crss_y_injury_model_comparison.csv'):.3f}; "
                   f"app form model {dep['y_injury'].test_roc_auc:.3f}"},
        {"task": "S2 Serious or fatal crash (CRSS)", "model": s2.model, "headline_metric": "ROC-AUC, 2024 test",
         "headline_value": f"{s2.test_roc_auc:.3f}",
         "detail": f"PR-AUC {s2.test_pr_auc:.3f} vs baseline {baseline_pr_auc(EVALUATION_DIR / 'crss_y_serious_model_comparison.csv'):.3f}; app form model {dep['y_serious'].test_roc_auc:.3f}"},
        {"task": "U1 Component routing (complaints)", "model": u1.model, "headline_metric": "Top-1 accuracy, 2024-26 test",
         "headline_value": f"{u1.test_top1_accuracy:.1%}",
         "detail": f"micro-F1 {u1.test_micro_f1:.3f}, macro-F1 {u1.test_macro_f1:.3f} across 19 groups"},
        {"task": "U2 Serious-incident flag (complaints)", "model": u2.model, "headline_metric": "ROC-AUC, 2024-26 test",
         "headline_value": f"{u2.test_roc_auc:.3f}",
         "detail": f"PR-AUC {u2.test_pr_auc:.3f}, recall {u2.test_recall:.2f}, precision {u2.test_precision:.2f}"},
    ]}
    (ART / "model_card.json").write_text(json.dumps(card, indent=1), encoding="utf-8")
    for f in sorted(ART.iterdir()):
        print(f"{f.name:32s} {f.stat().st_size / 1e6:8.1f} MB")
    print(json.dumps(card, indent=1))


if __name__ == "__main__":
    main()
