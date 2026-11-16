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


def fit_cluster(formula, data):
    fit = smf.ols(formula, data=data).fit()
    rob = fit.get_robustcov_results(cov_type="cluster", groups=data["Sample_ID"])
    rhs = formula.split("~", 1)[1]
    design = patsy.dmatrix(rhs, data, return_type="dataframe")
    if list(design.columns) != list(fit.model.exog_names):
        raise RuntimeError("Patsy design columns do not match Statsmodels coefficient order.")
    return fit, rob, design.design_info


def design_row(di, new):
    X = patsy.build_design_matrices([di], new, return_type="dataframe")[0]
    return np.asarray(X.iloc[0], dtype=float)


def estimate(rob, x):
    b = np.asarray(rob.params)
    V = np.asarray(rob.cov_params())
    est = float(x @ b)
    se = float(np.sqrt(max(x @ V @ x, 0)))
    return est, se, est - 1.96 * se, est + 1.96 * se


def predict(di, rob, new):
    return estimate(rob, design_row(di, new))


def contrast(di, rob, a, b):
    return estimate(rob, design_row(di, a) - design_row(di, b))


def omnibus(rob, required, label):
    names = list(rob.model.exog_names)
    idx = [i for i, name in enumerate(names) if all(token in name for token in required)]
    if not idx:
        return {"Test": label, "df": np.nan, "Wald_chi2": np.nan, "p_value": np.nan}
    R = np.zeros((len(idx), len(names)))
    for r, i in enumerate(idx):
        R[r, i] = 1.0
    w = rob.wald_test(R, scalar=True)
    return {"Test": label, "df": len(idx), "Wald_chi2": float(w.statistic), "p_value": float(w.pvalue)}


def save_info(name, fit, n_id):
    pd.DataFrame([{
        "Model": name, "Rows": int(fit.nobs), "Unique_ID": int(n_id),
        "AIC": fit.aic, "BIC": fit.bic, "R2": fit.rsquared,
        "Adjusted_R2": fit.rsquared_adj, "Covariance": "Cluster-robust by Sample_ID"
    }]).to_csv(c.RECOVERY_DIR / f"{name}_Model_Info.csv", index=False)


wald = []

# E1 Severity × Time
x = df.dropna(subset=["Recovery", "Severity", "Time", "Sample_ID"]).copy()
formula = "Recovery ~ C(Severity, Treatment(reference='Low')) * C(Time, Treatment(reference='T1'))"
fit, rob, di = fit_cluster(formula, x)
save_info("E1_Severity_Time", fit, x.Sample_ID.nunique())
rows = []
for s in c.SEVERITIES:
    for t in c.TIMES:
        rows.append([s, t, *predict(di, rob, pd.DataFrame({"Severity": [s], "Time": [t]}))])
pd.DataFrame(rows, columns=["Severity", "Time", "Estimated_Mean", "SE", "CI_Low", "CI_High"]).to_csv(c.RECOVERY_DIR / "E1_Severity_Time_Predictions.csv", index=False)
rows = []
for t in c.TIMES:
    rows.append([t, *contrast(di, rob,
        pd.DataFrame({"Severity": ["Low"], "Time": [t]}),
        pd.DataFrame({"Severity": ["High"], "Time": [t]}))])
pd.DataFrame(rows, columns=["Time", "Low_minus_High", "SE", "CI_Low", "CI_High"]).to_csv(c.RECOVERY_DIR / "E1_Low_High_Contrasts.csv", index=False)
wald.append(omnibus(rob, ["C(Severity", "C(Time", ":"], "Severity × Time"))

# E2 Severity × Landform × Time
x = df.dropna(subset=["Recovery", "Severity", "Landform", "Time", "Sample_ID"]).copy()
formula = "Recovery ~ C(Severity, Treatment(reference='Low')) * C(Landform, Treatment(reference='LwSlp')) * C(Time, Treatment(reference='T1'))"
fit, rob, di = fit_cluster(formula, x)
save_info("E2_Severity_Landform_Time", fit, x.Sample_ID.nunique())
rows = []
for s in c.SEVERITIES:
    for l in c.LANDFORMS:
        for t in c.TIMES:
            rows.append([s, l, t, *predict(di, rob, pd.DataFrame({"Severity": [s], "Landform": [l], "Time": [t]}))])
pd.DataFrame(rows, columns=["Severity", "Landform", "Time", "Estimated_Mean", "SE", "CI_Low", "CI_High"]).to_csv(c.RECOVERY_DIR / "E2_Landform_Predictions.csv", index=False)
wald.append(omnibus(rob, ["C(Severity", "C(Landform", "C(Time"], "Severity × Landform × Time"))

# E3 Severity × Vegetation × Time
x = df[df.Vegetation.isin(c.VEGETATION)].dropna(subset=["Recovery", "Severity", "Vegetation", "Time", "Sample_ID"]).copy()
formula = "Recovery ~ C(Severity, Treatment(reference='Low')) * C(Vegetation, Treatment(reference='EFWL')) * C(Time, Treatment(reference='T1'))"
fit, rob, di = fit_cluster(formula, x)
save_info("E3_Severity_Vegetation_Time", fit, x.Sample_ID.nunique())
rows = []
for s in c.SEVERITIES:
    for v in c.VEGETATION:
        for t in c.TIMES:
            rows.append([s, v, t, *predict(di, rob, pd.DataFrame({"Severity": [s], "Vegetation": [v], "Time": [t]}))])
pd.DataFrame(rows, columns=["Severity", "Vegetation", "Time", "Estimated_Mean", "SE", "CI_Low", "CI_High"]).to_csv(c.RECOVERY_DIR / "E3_Vegetation_Predictions.csv", index=False)
wald.append(omnibus(rob, ["C(Severity", "C(Vegetation", "C(Time"], "Severity × Vegetation × Time"))

# E4 integrated EFWL + SHMW
x = df[df.Vegetation.isin(c.INTEGRATED_VEGETATION)].dropna(subset=["Recovery", "Severity", "Landform", "Vegetation", "Time", "Sample_ID"]).copy()
formula = "Recovery ~ C(Severity, Treatment(reference='Low')) * C(Landform, Treatment(reference='LwSlp')) * C(Vegetation, Treatment(reference='EFWL')) * C(Time, Treatment(reference='T1'))"
fit, rob, di = fit_cluster(formula, x)
save_info("E4_Integrated", fit, x.Sample_ID.nunique())
rows = []
for s in c.SEVERITIES:
    for v in c.INTEGRATED_VEGETATION:
        for l in c.LANDFORMS:
            for t in c.TIMES:
                rows.append([s, v, l, t, *predict(di, rob, pd.DataFrame({"Severity": [s], "Vegetation": [v], "Landform": [l], "Time": [t]}))])
pd.DataFrame(rows, columns=["Severity", "Vegetation", "Landform", "Time", "Estimated_Mean", "SE", "CI_Low", "CI_High"]).to_csv(c.RECOVERY_DIR / "E4_Integrated_Predictions.csv", index=False)
wald.append(omnibus(rob, ["C(Severity", "C(Landform", "C(Vegetation", "C(Time"], "Severity × Landform × Vegetation × Time"))

wald_df = pd.DataFrame(wald)
wald_df.to_csv(c.RECOVERY_DIR / "Primary_Omnibus_Tests.csv", index=False)
print("=" * 72)
print("RECOVERY MODELS V2 COMPLETE")
print("=" * 72)
print(wald_df.to_string(index=False))
print(f"\nOutputs: {c.RECOVERY_DIR}")
