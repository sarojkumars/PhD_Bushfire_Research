import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf

spec = importlib.util.spec_from_file_location("config", Path(__file__).with_name("00_Config_v2.py"))
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
df = pd.read_csv(c.ANALYSIS_READY)

d = df[df["Vegetation"] == c.PRIMARY_VEGETATION].dropna(subset=["Recovery", "Severity", "Time", "Landform", "CHM", "Sample_ID"]).copy()
unique = d.drop_duplicates("Sample_ID").copy()

if d.groupby("Sample_ID")["CHM"].nunique().max() > 1:
    print("WARNING: CHM varies within at least one Sample_ID. Check whether CHM is truly static/pre-fire.")

q = unique["CHM"].quantile([0.25, 0.50, 0.75])
levels = {"P25": float(q.loc[0.25]), "P50": float(q.loc[0.50]), "P75": float(q.loc[0.75])}
chm_mean = d["CHM"].mean(); chm_sd = d["CHM"].std(ddof=0)
d["CHM_z"] = (d["CHM"] - chm_mean) / chm_sd
weights = unique["Landform"].value_counts(normalize=True).reindex(c.LANDFORMS).fillna(0.0)

formula = (
    "Recovery ~ C(Severity, Treatment(reference='Low')) * C(Time, Treatment(reference='T1'))"
    " + CHM_z * C(Severity, Treatment(reference='Low')) * C(Time, Treatment(reference='T1'))"
    " + C(Landform, Treatment(reference='LwSlp'))"
)
fit = smf.ols(formula, data=d).fit()
rob = fit.get_robustcov_results(cov_type="cluster", groups=d["Sample_ID"])
rhs = formula.split("~", 1)[1]
design = patsy.dmatrix(rhs, d, return_type="dataframe")
if list(design.columns) != list(fit.model.exog_names):
    raise RuntimeError("CHM design matrix column order mismatch.")
di = design.design_info
b = np.asarray(rob.params); V = np.asarray(rob.cov_params())


def avg_design(severity, time, chm):
    rows, ww = [], []
    for land in c.LANDFORMS:
        w = float(weights.loc[land])
        if w <= 0: continue
        rows.append({"Severity": severity, "Time": time, "Landform": land, "CHM_z": (chm - chm_mean) / chm_sd})
        ww.append(w)
    X = patsy.build_design_matrices([di], pd.DataFrame(rows), return_type="dataframe")[0]
    ww = np.asarray(ww, float); ww /= ww.sum()
    return np.average(np.asarray(X), axis=0, weights=ww)


def est(x):
    estimate = float(x @ b); se = float(np.sqrt(max(x @ V @ x, 0)))
    return estimate, se, estimate - 1.96 * se, estimate + 1.96 * se

rows = []
for sev in c.SEVERITIES:
    for t in c.TIMES:
        for label, chm in levels.items():
            rows.append([sev, t, label, chm, *est(avg_design(sev, t, chm))])
pd.DataFrame(rows, columns=["Severity", "Time", "CHM_Level", "CHM_m", "Estimated_Recovery", "SE", "CI_Low", "CI_High"]).to_csv(c.CHM_DIR / "CHM_Percentile_Predictions.csv", index=False)

rows = []
for sev in c.SEVERITIES:
    for t in c.TIMES:
        x50 = avg_design(sev, t, levels["P50"])
        for label in ["P25", "P75"]:
            rows.append([sev, t, label, levels[label], *est(avg_design(sev, t, levels[label]) - x50)])
pd.DataFrame(rows, columns=["Severity", "Time", "CHM_Level", "CHM_m", "Delta_vs_P50", "SE", "CI_Low", "CI_High"]).to_csv(c.CHM_DIR / "CHM_Contrasts_vs_P50.csv", index=False)

pd.DataFrame([{
    "Rows": len(d), "Unique_ID": d.Sample_ID.nunique(),
    "CHM_P25_m": levels["P25"], "CHM_P50_m": levels["P50"], "CHM_P75_m": levels["P75"],
    "AIC": fit.aic, "BIC": fit.bic, "R2": fit.rsquared, "Adjusted_R2": fit.rsquared_adj,
    "Covariance": "Cluster-robust by Sample_ID"
}]).to_csv(c.CHM_DIR / "CHM_Linear_Model_Info.csv", index=False)

if c.RUN_GAM:
    try:
        from pygam import LinearGAM, s, f
        land_code = {land: i for i, land in enumerate(c.LANDFORMS)}
        grid = np.linspace(float(unique.CHM.quantile(0.05)), float(unique.CHM.quantile(0.95)), 200)
        pred_rows, info_rows, deriv_rows = [], [], []
        for sev in c.SEVERITIES:
            for t in c.TIMES:
                g = d[(d.Severity == sev) & (d.Time == t)].copy()
                g["LandCode"] = g["Landform"].map(land_code)
                X = g[["CHM", "LandCode"]].to_numpy(); y = g["Recovery"].to_numpy()
                gam = LinearGAM(s(0, n_splines=10) + f(1)).gridsearch(X, y, lam=np.logspace(-3, 3, 13), progress=False)
                curve = []
                for chm in grid:
                    vals, ww = [], []
                    for land in c.LANDFORMS:
                        w = float(weights.loc[land])
                        if w <= 0: continue
                        vals.append(float(gam.predict(np.array([[chm, land_code[land]]]))[0])); ww.append(w)
                    curve.append(np.average(vals, weights=ww))
                curve = np.asarray(curve); derivative = np.gradient(curve, grid)
                for chm, yhat, dy in zip(grid, curve, derivative): pred_rows.append([sev, t, chm, yhat, dy])
                idx = int(np.nanargmax(np.abs(derivative)))
                deriv_rows.append({"Severity": sev, "Time": t, "CHM_at_max_abs_derivative": grid[idx], "Derivative": derivative[idx]})
                stats = gam.statistics_; pseudo = stats.get("pseudo_r2", np.nan)
                if isinstance(pseudo, dict): pseudo = pseudo.get("explained_deviance", np.nan)
                info_rows.append({"Severity": sev, "Time": t, "EDOF_Total": stats.get("edof", np.nan), "Pseudo_R2": pseudo, "AIC": stats.get("AIC", np.nan), "GCV": stats.get("GCV", np.nan)})
        pd.DataFrame(pred_rows, columns=["Severity", "Time", "CHM_m", "Estimated_Recovery", "Derivative"]).to_csv(c.CHM_DIR / "CHM_GAM_Predictions.csv", index=False)
        pd.DataFrame(info_rows).to_csv(c.CHM_DIR / "CHM_GAM_Model_Info.csv", index=False)
        pd.DataFrame(deriv_rows).to_csv(c.SUPP_CHM_DIR / "S_CHM_GAM_Max_Derivative.csv", index=False)
    except ImportError:
        print("pyGAM is not installed. Linear CHM analysis completed. Optional: pip install pygam")

print("=" * 72)
print("CHM ANALYSIS V2 COMPLETE")
print("=" * 72)
print(f"CHM P25/P50/P75 = {levels['P25']:.2f}, {levels['P50']:.2f}, {levels['P75']:.2f} m")
print(f"Outputs: {c.CHM_DIR}")
