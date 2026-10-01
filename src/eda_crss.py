"""Exploratory analysis and statistical tests for the CRSS crash table (training years only
for associations, all years for trends). Writes tables to outputs/tables/eda_crss_* and
figures to outputs/figures/eda_crss_*.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from project_paths import CRSS_TRAIN_YEARS, CRSS_YEARS, FIGURE_DIR, PROCESSED_DIR, TABLE_DIR, crss_extracted_dir

TABLES = ["accident", "vehicle", "person", "crashrf", "weather", "cevent", "damage", "distract", "drimpair",
          "driverrf", "factor", "maneuver", "vehiclesf", "violatn", "vision", "vevent", "vsoe", "parkwork",
          "pvehiclesf", "nmcrash", "nmdistract", "nmimpair", "nmprior", "personrf", "safetyeq", "pbtype"]


def code_labels() -> dict:
    """(table, field, code) -> official label, from the NAME columns of the 2022 files."""
    labels = {}
    for t in TABLES:
        df = pd.read_csv(crss_extracted_dir(2022) / f"{t}.csv", encoding="latin-1", low_memory=False, nrows=200000)
        df.columns = [c.upper() for c in df.columns]
        for c in df.columns:
            if c + "NAME" in df.columns:
                pairs = df[[c, c + "NAME"]].drop_duplicates()
                for code, name in pairs.itertuples(index=False):
                    labels[(t, c, str(code))] = str(name)
    return labels


def describe(feature: str, labels: dict) -> str:
    table, rest = feature.split("__", 1)
    table = "accident" if table == "acc" else table
    if "=" in rest:
        field, code = rest.split("=")
        code = code[:-2] if code.endswith(".0") else code
        return f"{table}.{field} = {labels.get((table, field, code), code)} (count)"
    return f"{table}.{rest}"


def cramers_v(table: pd.DataFrame) -> tuple[float, float]:
    chi2, p, _, _ = stats.chi2_contingency(table)
    n = table.values.sum()
    return float(np.sqrt(chi2 / (n * (min(table.shape) - 1)))), float(p)


def main() -> None:
    df = pd.read_parquet(PROCESSED_DIR / "crss_crash_features.parquet")
    reg = pd.read_csv(TABLE_DIR / "crss_feature_register.csv").set_index("feature")
    labels = code_labels()
    feats = [c for c in df.columns if "__" in c]
    train = df[df["YEAR"].isin(CRSS_TRAIN_YEARS)]
    results = {}

    # 1. Outcome prevalence: raw sample vs survey-weighted national estimate, by year.
    prev = []
    for y, d in df.groupby("YEAR"):
        for t in ["y_injury", "y_serious"]:
            m = d[t] >= 0
            prev.append({"year": y, "target": t, "sample_rate": d.loc[m, t].mean(),
                         "weighted_national_rate": np.average(d.loc[m, t], weights=d.loc[m, "WEIGHT"]),
                         "weighted_crashes_millions": d.loc[m, "WEIGHT"].sum() / 1e6})
    prev = pd.DataFrame(prev)
    prev.to_csv(TABLE_DIR / "eda_crss_prevalence.csv", index=False)

    # 2. Temporal drift test: is the injury rate the same in every year? (chi-square)
    for t in ["y_injury", "y_serious"]:
        m = df[t] >= 0
        v, p = cramers_v(pd.crosstab(df.loc[m, "YEAR"], df.loc[m, t]))
        results[f"year_vs_{t}"] = {"cramers_v": v, "p_value": p}

    # 3. Univariate screening of every feature (training years): ROC-AUC of the raw feature
    #    (direction-free) plus Spearman correlation. Doubles as a leakage check: no single
    #    legitimate field should separate the classes almost perfectly.
    rows = []
    for t in ["y_injury", "y_serious"]:
        m = train[t] >= 0
        y = train.loc[m, t].values
        for f in feats:
            x = train.loc[m, f]
            if reg.loc[f, "kind"] == "categorical_code":
                # one-vs-rest per code is covered by Cramér's V below; AUC on target-mean encoding
                x = x.fillna(-9)
                x = x.map(pd.Series(y, index=x.index).groupby(x).mean())
            x = x.fillna(x.median() if x.notna().any() else 0)
            if x.nunique() < 2:
                continue
            auc = roc_auc_score(y, x)
            rows.append({"target": t, "feature": f, "description": describe(f, labels),
                         "group": reg.loc[f, "group"], "auc": max(auc, 1 - auc),
                         "direction": "higher -> more likely" if auc >= 0.5 else "higher -> less likely",
                         "nonzero_share": float((x != 0).mean())})
    screen = pd.DataFrame(rows).sort_values(["target", "auc"], ascending=[True, False])
    screen.to_csv(TABLE_DIR / "eda_crss_univariate_screen.csv", index=False)
    results["leakage_check_max_single_feature_auc"] = screen.groupby("target")["auc"].max().to_dict()

    # 4. Chi-square / Cramér's V for every crash-level categorical field.
    cv = []
    for f in [f for f in feats if reg.loc[f, "kind"] == "categorical_code"]:
        for t in ["y_injury", "y_serious"]:
            m = train[t] >= 0
            v, p = cramers_v(pd.crosstab(train.loc[m, f].fillna(-9), train.loc[m, t]))
            cv.append({"field": f.split("__")[1], "target": t, "cramers_v": v, "p_value": p})
    pd.DataFrame(cv).sort_values(["target", "cramers_v"], ascending=[True, False]).to_csv(
        TABLE_DIR / "eda_crss_cramers_v.csv", index=False)

    # 5. Numeric comparisons (Mann-Whitney U): driver age, travel speed, speed limit, model year.
    mw = []
    for f in ["person__AGE_IM__min", "person__AGE_IM__max", "vehicle__TRAV_SP__max", "vehicle__VSPD_LIM__max",
              "vehicle__MDLYR_IM__min", "acc__VE_TOTAL", "acc__PERMVIT"]:
        if f not in train:
            continue
        for t in ["y_injury", "y_serious"]:
            m = (train[t] >= 0) & train[f].notna()
            a, b = train.loc[m & (train[t] == 1), f], train.loc[m & (train[t] == 0), f]
            u, p = stats.mannwhitneyu(a, b)
            mw.append({"feature": describe(f, labels), "target": t, "median_positive": a.median(),
                       "median_negative": b.median(), "rank_biserial": 1 - 2 * u / (len(a) * len(b)), "p_value": p,
                       "n": int(m.sum())})
    pd.DataFrame(mw).to_csv(TABLE_DIR / "eda_crss_mannwhitney.csv", index=False)

    # 6. Vehicle-defect finding: are crashes with a police-recorded vehicle defect more severe?
    #    Read from the raw factor table: the feature table keeps only frequent codes.
    defect_keys = set()
    for y in CRSS_YEARS:
        f = pd.read_csv(crss_extracted_dir(y) / "factor.csv", encoding="latin-1", usecols=["CASENUM", "VEHICLECC"])
        f = f[~f["VEHICLECC"].isin([0, 98, 99])]  # None noted / no details / unknown
        defect_keys |= {(y, c) for c in f["CASENUM"].unique()}
    has_def = pd.Series([k in defect_keys for k in zip(df["YEAR"], df["CASENUM"])], index=df.index)
    for t in ["y_injury", "y_serious"]:
        m = df[t] >= 0
        tab = pd.crosstab(has_def[m], df.loc[m, t])
        chi2, p, _, _ = stats.chi2_contingency(tab)
        rates = df.loc[m].groupby(has_def[m])[t].mean()
        results[f"vehicle_defect_vs_{t}"] = {"rate_with_defect": float(rates.get(True)),
                                             "rate_without_defect": float(rates.get(False)),
                                             "crashes_with_defect": int(has_def[m].sum()), "chi2_p": float(p)}

    pd.Series(results).to_json(TABLE_DIR / "eda_crss_tests.json", indent=2)

    # Figures
    fig, ax = plt.subplots(figsize=(7, 4))
    for t, lab in [("y_injury", "Injury crash"), ("y_serious", "Serious/fatal crash")]:
        d = prev[prev.target == t]
        ax.plot(d.year, d.weighted_national_rate * 100, marker="o", label=f"{lab} (weighted)")
        ax.plot(d.year, d.sample_rate * 100, marker="o", ls="--", alpha=.6, label=f"{lab} (sample)")
    ax.set_ylabel("% of police-reported crashes"); ax.set_xticks(list(CRSS_YEARS)); ax.legend(fontsize=8)
    ax.set_title("Crash severity outcomes, CRSS 2020-2024"); fig.tight_layout()
    fig.savefig(FIGURE_DIR / "eda_crss_prevalence.png", dpi=150); plt.close(fig)

    for t in ["y_injury", "y_serious"]:
        top = screen[(screen.target == t)].head(20).iloc[::-1]
        fig, ax = plt.subplots(figsize=(9, 7))
        colors = ["#c0504d" if g == "post_crash" else "#4f81bd" for g in top.group]
        ax.barh(top.description.str.slice(0, 70), top.auc, color=colors)
        ax.axvline(0.5, color="grey", lw=.8); ax.set_xlim(0.5, max(0.8, top.auc.max() + .02))
        ax.set_xlabel("Single-feature ROC-AUC (training years)")
        ax.set_title(f"Strongest individual associations: {t} (red = post-crash field)")
        fig.tight_layout(); fig.savefig(FIGURE_DIR / f"eda_crss_top_{t}.png", dpi=150); plt.close(fig)

    print(pd.Series(results).to_string())
    for t in ["y_injury", "y_serious"]:
        print(f"\nTop 25 for {t}")
        print(screen[screen.target == t].head(25)[["description", "group", "auc", "direction"]].to_string(index=False))
    print(prev.round(4).to_string(index=False))
    print(pd.read_csv(TABLE_DIR / "eda_crss_mannwhitney.csv").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
