"""Exploratory analysis of owner complaints, and the CRSS-complaint linkage analysis.

Linkage: CRSS survey weights estimate how many vehicles of each make/model-year were
involved in police-reported crashes nationally (an on-road exposure proxy). Dividing
complaint counts by that estimate gives a complaint rate per exposure, which neither
dataset can provide alone. Both sides are restricted to the same calendar window (2020-2024).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import CountVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from project_paths import CRSS_YEARS, FIGURE_DIR, PROCESSED_DIR, TABLE_DIR, crss_extracted_dir

RNG = 42


def norm(s: pd.Series) -> pd.Series:
    return s.astype("string").str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)


def log_odds_terms(texts: pd.Series, y: pd.Series, top: int = 25) -> pd.DataFrame:
    """Informative-Dirichlet-prior log-odds (Monroe et al., 2008) for terms in y=1 vs y=0."""
    cv = CountVectorizer(ngram_range=(1, 2), min_df=20, stop_words="english", token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
    X = cv.fit_transform(texts)
    a = np.asarray(X[y.values == 1].sum(0)).ravel(); b = np.asarray(X[y.values == 0].sum(0)).ravel()
    prior = (a + b) * 0.01 + 0.01; a0, b0, p0 = a.sum(), b.sum(), prior.sum()
    delta = np.log((a + prior) / (a0 + p0 - a - prior)) - np.log((b + prior) / (b0 + p0 - b - prior))
    z = delta / np.sqrt(1 / (a + prior) + 1 / (b + prior))
    terms = np.array(cv.get_feature_names_out())
    idx = np.argsort(-z)[:top]
    return pd.DataFrame({"term": terms[idx], "z": z[idx], "count_in_class": a[idx]})


def main() -> None:
    c = pd.read_parquet(PROCESSED_DIR / "complaints_model.parquet",
                        columns=["ODINO", "date", "year", "split", "make", "model", "model_year", "text", "components",
                                 "crash", "fire", "injured", "deaths", "y_serious"])
    comp_cols = [x for x in pd.read_parquet(PROCESSED_DIR / "complaints_model.parquet").columns if x.startswith("c__")]
    cc = pd.read_parquet(PROCESSED_DIR / "complaints_model.parquet", columns=["year"] + comp_cols)
    results = {}

    # 1. Volume, seriousness and component mix over time.
    yearly = c.groupby("year").agg(complaints=("ODINO", "size"), serious_rate=("y_serious", "mean"),
                                   crash_rate=("crash", "mean"), fire_rate=("fire", "mean"),
                                   injury_rate=("injured", "mean"))
    yearly.to_csv(TABLE_DIR / "eda_complaints_yearly.csv")
    recent = c[c.year >= 2010]
    tau, p = stats.kendalltau(yearly.loc[2010:].index, yearly.loc[2010:, "serious_rate"])
    results["serious_rate_trend_2010_on"] = {"kendall_tau": tau, "p_value": p}
    mix = cc[cc.year >= 2010].groupby("year")[comp_cols].mean()
    mix.columns = [x[3:] for x in mix.columns]
    mix.to_csv(TABLE_DIR / "eda_complaints_component_mix.csv")
    trend = {col: stats.kendalltau(mix.index, mix[col]) for col in mix.columns}
    trend = pd.DataFrame({k: {"kendall_tau": v[0], "p_value": v[1], "share_2010": mix[k].iloc[0],
                              "share_latest": mix[k].iloc[-1]} for k, v in trend.items()}).T.sort_values("kendall_tau")
    trend.to_csv(TABLE_DIR / "eda_complaints_component_trends.csv")

    # 2. Narrative length vs seriousness (Mann-Whitney U).
    words = recent["text"].str.split().str.len()
    u, p = stats.mannwhitneyu(words[recent.y_serious == 1], words[recent.y_serious == 0])
    results["narrative_words_serious_vs_not"] = {
        "median_serious": float(words[recent.y_serious == 1].median()),
        "median_not": float(words[recent.y_serious == 0].median()),
        "rank_biserial": float(1 - 2 * u / ((recent.y_serious == 1).sum() * (recent.y_serious == 0).sum())), "p_value": float(p)}

    # 3. Which components are most associated with serious outcomes? (chi-square per component)
    rows = []
    cs = pd.read_parquet(PROCESSED_DIR / "complaints_model.parquet", columns=["year", "y_serious"] + comp_cols)
    cs = cs[cs.year >= 2010]
    for col in comp_cols:
        tab = pd.crosstab(cs[col], cs.y_serious)
        chi2, p, _, _ = stats.chi2_contingency(tab)
        rows.append({"component": col[3:], "complaints": int(cs[col].sum()),
                     "serious_rate_if_component": cs.loc[cs[col] == 1, "y_serious"].mean(),
                     "serious_rate_otherwise": cs.loc[cs[col] == 0, "y_serious"].mean(), "chi2_p": p})
    sev = pd.DataFrame(rows).assign(relative_risk=lambda d: d.serious_rate_if_component / d.serious_rate_otherwise)
    sev.sort_values("relative_risk", ascending=False).to_csv(TABLE_DIR / "eda_complaints_component_severity.csv", index=False)

    # 4. Distinctive language of serious complaints (training period sample only).
    sample = c[(c.split == "train") & (c.year >= 2010)].sample(200000, random_state=RNG)
    log_odds_terms(sample.text, sample.y_serious).to_csv(TABLE_DIR / "eda_complaints_serious_terms.csv", index=False)

    # 5. Linkage: complaint rate per CRSS-estimated crash exposure, by make and model year.
    veh = []
    for y in CRSS_YEARS:
        v = pd.read_csv(crss_extracted_dir(y) / "vehicle.csv", encoding="latin-1", low_memory=False,
                        usecols=["CASENUM", "MOD_YEAR", "VPICMAKENAME", "WEIGHT"])
        veh.append(v)
    veh = pd.concat(veh)
    veh = veh[veh.MOD_YEAR.between(2000, 2025)]
    veh["mk"] = norm(veh.VPICMAKENAME)
    exposure = veh.groupby(["mk", "MOD_YEAR"]).agg(crss_sample_vehicles=("WEIGHT", "size"),
                                                   est_crash_involved_vehicles=("WEIGHT", "sum"))
    cw = c[c.year.between(2020, 2024)].assign(mk=norm(c["make"]), my=pd.to_numeric(c["model_year"], errors="coerce"))
    cnt = cw.groupby(["mk", "my"]).agg(complaints=("ODINO", "size"), serious_complaints=("y_serious", "sum"))
    cnt.index.names = ["mk", "MOD_YEAR"]
    link = exposure.join(cnt, how="inner")
    link = link[link.crss_sample_vehicles >= 30]
    link["complaints_per_10k_crash_involved"] = link.complaints / link.est_crash_involved_vehicles * 1e4
    link.sort_values("complaints_per_10k_crash_involved", ascending=False).to_csv(TABLE_DIR / "eda_linkage_make_modelyear.csv")
    rho, p = stats.spearmanr(link.est_crash_involved_vehicles, link.complaints)
    results["linkage"] = {"make_modelyear_cells": int(len(link)), "spearman_exposure_vs_complaints": rho, "p_value": p,
                          "complaint_rate_p90_over_p10": float(link.complaints_per_10k_crash_involved.quantile(.9) /
                                                               link.complaints_per_10k_crash_involved.quantile(.1))}
    by_make = link.groupby(level="mk").agg(complaints=("complaints", "sum"), exposure=("est_crash_involved_vehicles", "sum"))
    by_make = by_make[by_make.exposure > 2e5]
    by_make["complaints_per_10k"] = by_make.complaints / by_make.exposure * 1e4
    by_make.sort_values("complaints_per_10k", ascending=False).to_csv(TABLE_DIR / "eda_linkage_make.csv")

    pd.Series(results).to_json(TABLE_DIR / "eda_complaints_tests.json", indent=2)

    # Figures
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    yearly.loc[1995:2025, "complaints"].plot(ax=ax[0], marker="o"); ax[0].set_title("Complaints received per year")
    (yearly.loc[1995:2025, ["serious_rate", "crash_rate", "injury_rate", "fire_rate"]] * 100).plot(ax=ax[1])
    ax[1].set_title("% of complaints reporting crash / injury / fire"); fig.tight_layout()
    fig.savefig(FIGURE_DIR / "eda_complaints_yearly.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    (mix[[col for col in mix.columns if trend.loc[col, "p_value"] < 0.01]].pipe(
        lambda m: m[m.iloc[-1].sort_values().index[-8:]]) * 100).plot(ax=ax)
    ax.set_ylabel("% of complaints"); ax.set_title("Component mix over time (largest components with significant trends)")
    fig.tight_layout(); fig.savefig(FIGURE_DIR / "eda_complaints_component_mix.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 6))
    top = by_make.sort_values("complaints_per_10k").tail(20)
    ax.barh(top.index, top.complaints_per_10k); ax.set_xlabel("complaints per 10,000 crash-involved vehicles (2020-2024)")
    ax.set_title("Complaint rate adjusted for on-road exposure (CRSS-weighted)"); fig.tight_layout()
    fig.savefig(FIGURE_DIR / "eda_linkage_make.png", dpi=150); plt.close(fig)

    print(pd.Series(results).to_string())
    print(trend.round(4).to_string())
    print(sev.sort_values("relative_risk", ascending=False).round(3).to_string(index=False))
    print(pd.read_csv(TABLE_DIR / "eda_complaints_serious_terms.csv").head(25).to_string(index=False))
    print(by_make.sort_values("complaints_per_10k", ascending=False).round(1).to_string())


if __name__ == "__main__":
    main()
