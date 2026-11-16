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

# -----------------------------
# 04. EFWL climate-stress Perturbation scenarios.. 
# -----------------------

d = df[df["Vegetation"] == c.PRIMARY_VEGETATION].dropna(
    subset=["Recovery", "Severity", "Time", "Landform", "RfSpr", "SMSpr", "TMaxSpr", "Sample_ID"]
).copy()

climate = ["RfSpr", "SMSpr", "TMaxSpr"]
means = d[climate].mean()
sds = d[climate].std(ddof=0)

for col in climate:
    d[col + "_z"] = (d[col] - means[col]) / sds[col]

q = d[climate].quantile([0.25, 0.50, 0.75])

SCENARIOS = {
    "Reference": {
        "RfSpr": float(q.loc[0.50, "RfSpr"]),
        "SMSpr": float(q.loc[0.50, "SMSpr"]),
        "TMaxSpr": float(q.loc[0.50, "TMaxSpr"])
    },
    "HotDry": {
        "RfSpr": float(q.loc[0.25, "RfSpr"]),
        "SMSpr": float(q.loc[0.25, "SMSpr"]),
        "TMaxSpr": float(q.loc[0.75, "TMaxSpr"])
    },
    "CoolWet": {
        "RfSpr": float(q.loc[0.75, "RfSpr"]),
        "SMSpr": float(q.loc[0.75, "SMSpr"]),
        "TMaxSpr": float(q.loc[0.25, "TMaxSpr"])
    }
}

pd.DataFrame(SCENARIOS).T.to_csv(c.CLIMATE_DIR / "Climate_Scenario_Values.csv")

weights = d.drop_duplicates("Sample_ID")["Landform"].value_counts(normalize=True).reindex(c.LANDFORMS).fillna(0.0)

# ================
# Climate model.. 
# ============================================================

formula = (
    "Recovery ~ C(Severity, Treatment(reference='Low')) * C(Time, Treatment(reference='T1'))"
    " + RfSpr_z * C(Severity, Treatment(reference='Low')) * C(Time, Treatment(reference='T1'))"
    " + SMSpr_z * C(Severity, Treatment(reference='Low')) * C(Time, Treatment(reference='T1'))"
    " + TMaxSpr_z * C(Severity, Treatment(reference='Low')) * C(Time, Treatment(reference='T1'))"
    " + C(Landform, Treatment(reference='LwSlp'))"
)

fit = smf.ols(formula, data=d).fit()
rob = fit.get_robustcov_results(cov_type="cluster", groups=d["Sample_ID"])

rhs = formula.split("~", 1)[1]
design = patsy.dmatrix(rhs, d, return_type="dataframe")

if list(design.columns) != list(fit.model.exog_names):
    raise RuntimeError("Climate design matrix column order mismatch.")

di = design.design_info
b = np.asarray(rob.params)
V = np.asarray(rob.cov_params())


def zvals(vals):
    return {
        "RfSpr_z": (vals["RfSpr"] - means["RfSpr"]) / sds["RfSpr"],
        "SMSpr_z": (vals["SMSpr"] - means["SMSpr"]) / sds["SMSpr"],
        "TMaxSpr_z": (vals["TMaxSpr"] - means["TMaxSpr"]) / sds["TMaxSpr"]
    }


def interpolate(a, b_, frac):
    return {col: a[col] + frac * (b_[col] - a[col]) for col in climate}


def avg_design(sev, t, vals):
    rows, ww = [], []

    for land in c.LANDFORMS:
        w = float(weights.loc[land])
        if w <= 0:
            continue

        rows.append({"Severity": sev, "Time": t, "Landform": land, **zvals(vals)})
        ww.append(w)

    X = patsy.build_design_matrices([di], pd.DataFrame(rows), return_type="dataframe")[0]
    ww = np.asarray(ww, float)
    ww /= ww.sum()

    return np.average(np.asarray(X), axis=0, weights=ww)


def est(x):
    e = float(x @ b)
    se = float(np.sqrt(max(x @ V @ x, 0)))
    return e, se, e - 1.96 * se, e + 1.96 * se


# =========================
# A. Progressive climatic pertubration.. 
# =====================================

rows = []

for sev in c.SEVERITIES:
    for t in c.TIMES:

        xref = avg_design(sev, t, SCENARIOS["Reference"])

        for pathway in ["HotDry", "CoolWet"]:
            for frac in [0, 0.25, 0.50, 0.75, 1.0]:

                vals = interpolate(SCENARIOS["Reference"], SCENARIOS[pathway], frac)
                x = avg_design(sev, t, vals)

                e, se, lo, hi = est(x)
                de, dse, dlo, dhi = est(x - xref)

                rows.append([
                    sev, t, pathway, frac,
                    vals["RfSpr"], vals["SMSpr"], vals["TMaxSpr"],
                    e, se, lo, hi,
                    de, dse, dlo, dhi
                ])

pd.DataFrame(
    rows,
    columns=[
        "Severity", "Time", "Pathway", "Progress",
        "Rainfall", "SoilMoisture", "Tmax",
        "Predicted_Recovery", "Recovery_SE", "Recovery_CI_Low", "Recovery_CI_High",
        "Delta_Recovery", "Delta_SE", "Delta_CI_Low", "Delta_CI_High"
    ]
).to_csv(c.CLIMATE_DIR / "Climate_Progressive_Perturbation.csv", index=False)


# =================================
# B. Severity-dependent climate sensitivity..
# High response minus Low response
# ========================================

rows = []

for t in c.TIMES:
    for pathway in ["HotDry", "CoolWet"]:

        dvec = (
            avg_design("High", t, SCENARIOS[pathway]) -
            avg_design("High", t, SCENARIOS["Reference"])
        ) - (
            avg_design("Low", t, SCENARIOS[pathway]) -
            avg_design("Low", t, SCENARIOS["Reference"])
        )

        rows.append([t, pathway, *est(dvec)])

pd.DataFrame(
    rows,
    columns=["Time", "Pathway", "High_minus_Low_Sensitivity", "SE", "CI_Low", "CI_High"]
).to_csv(c.CLIMATE_DIR / "Climate_Severity_Sensitivity_Difference.csv", index=False)


# ============================================================
# Primary model info.. 
# ============================================================

pd.DataFrame([{
    "Rows": len(d),
    "Unique_ID": d.Sample_ID.nunique(),
    "AIC": fit.aic,
    "BIC": fit.bic,
    "R2": fit.rsquared,
    "Adjusted_R2": fit.rsquared_adj,
    "Covariance": "Cluster-robust by Sample_ID"
}]).to_csv(c.CLIMATE_DIR / "Climate_Primary_Model_Info.csv", index=False)


# ===========================
# C. LANDFORM-CONDITIONED climate sensitivity.. 
# ======================================

land_rows, comparison_rows = [], []

for t in c.TIMES:

    g = d[d.Time == t].copy()

    base_formula = (
        "Recovery ~ C(Severity, Treatment(reference='Low'))"
        " + C(Landform, Treatment(reference='LwSlp'))"
        " + RfSpr_z * C(Severity, Treatment(reference='Low'))"
        " + SMSpr_z * C(Severity, Treatment(reference='Low'))"
        " + TMaxSpr_z * C(Severity, Treatment(reference='Low'))"
    )

    ext_formula = (
        "Recovery ~ C(Severity, Treatment(reference='Low'))"
        " + C(Landform, Treatment(reference='LwSlp'))"
        " + RfSpr_z * C(Severity, Treatment(reference='Low')) * C(Landform, Treatment(reference='LwSlp'))"
        " + SMSpr_z * C(Severity, Treatment(reference='Low')) * C(Landform, Treatment(reference='LwSlp'))"
        " + TMaxSpr_z * C(Severity, Treatment(reference='Low')) * C(Landform, Treatment(reference='LwSlp'))"
    )

    base_fit = smf.ols(base_formula, data=g).fit()
    ext_fit = smf.ols(ext_formula, data=g).fit()
    ext_rob = ext_fit.get_robustcov_results(cov_type="HC3")

    dm = patsy.dmatrix(ext_formula.split("~", 1)[1], g, return_type="dataframe")

    if list(dm.columns) != list(ext_fit.model.exog_names):
        raise RuntimeError(f"Landform climate design mismatch at {t}.")

    edi = dm.design_info
    eb = np.asarray(ext_rob.params)
    eV = np.asarray(ext_rob.cov_params())

    comparison_rows.append({
        "Time": t,
        "Base_AIC": base_fit.aic,
        "Extended_AIC": ext_fit.aic,
        "Delta_AIC_BaseMinusExtended": base_fit.aic - ext_fit.aic,
        "Base_Adjusted_R2": base_fit.rsquared_adj,
        "Extended_Adjusted_R2": ext_fit.rsquared_adj,
        "Delta_Adjusted_R2": ext_fit.rsquared_adj - base_fit.rsquared_adj
    })

    def x_land(sev, land, vals):
        X = patsy.build_design_matrices(
            [edi],
            pd.DataFrame([{"Severity": sev, "Landform": land, **zvals(vals)}]),
            return_type="dataframe"
        )[0]

        return np.asarray(X.iloc[0], float)

    def eest(x):
        e = float(x @ eb)
        se = float(np.sqrt(max(x @ eV @ x, 0)))
        return e, se, e - 1.96 * se, e + 1.96 * se

    for sev in c.SEVERITIES:
        for land in c.LANDFORMS:

            xr = x_land(sev, land, SCENARIOS["Reference"])

            for pathway in ["HotDry", "CoolWet"]:
                dx = x_land(sev, land, SCENARIOS[pathway]) - xr
                land_rows.append([sev, t, land, pathway, *eest(dx)])


pd.DataFrame(
    land_rows,
    columns=["Severity", "Time", "Landform", "Pathway", "Delta_Recovery", "SE", "CI_Low", "CI_High"]
).to_csv(c.CLIMATE_DIR / "Climate_Landform_Endpoint_Contrasts.csv", index=False)

pd.DataFrame(comparison_rows).to_csv(
    c.CLIMATE_DIR / "Climate_Landform_Model_Comparison.csv",
    index=False
)


# =============================
# D. JOINT TMAX × SOIL-MOISTURE response surface
# Rainfall @ P50; Tmax and SM varied empirically P10–P90
# ============================================================

surface_times = ["T1", "T3", "T5"]
n_grid = 50

rf_ref = float(d["RfSpr"].quantile(0.50))
sm_grid = np.linspace(d["SMSpr"].quantile(0.10), d["SMSpr"].quantile(0.90), n_grid)
tmax_grid = np.linspace(d["TMaxSpr"].quantile(0.10), d["TMaxSpr"].quantile(0.90), n_grid)

surface_rows = []

for sev in c.SEVERITIES:
    for t in surface_times:

        xref = avg_design(sev, t, SCENARIOS["Reference"])

        for sm in sm_grid:
            for tmax in tmax_grid:

                vals = {
                    "RfSpr": rf_ref,
                    "SMSpr": float(sm),
                    "TMaxSpr": float(tmax)
                }

                x = avg_design(sev, t, vals)

                e, se, lo, hi = est(x)
                de, dse, dlo, dhi = est(x - xref)

                surface_rows.append([
                    sev, t, rf_ref, sm, tmax,
                    e, se, lo, hi,
                    de, dse, dlo, dhi
                ])


surface = pd.DataFrame(
    surface_rows,
    columns=[
        "Severity", "Time", "Rainfall", "SoilMoisture", "Tmax",
        "Predicted_Recovery", "Predicted_SE", "Predicted_CI_Low", "Predicted_CI_High",
        "Delta_Recovery", "Delta_SE", "Delta_CI_Low", "Delta_CI_High"
    ]
)

surface.to_csv(
    c.CLIMATE_DIR / "Climate_Response_Surface.csv",
    index=False
)


# ===========
# E. Generating HIGH–LOW Climate vulnerability surface.. 
# Difference-of-differences:
# (High altered climate - High reference)
# -
# (Low altered climate - Low reference)
# ====

vulnerability_rows = []

for t in surface_times:

    x_high_ref = avg_design("High", t, SCENARIOS["Reference"])
    x_low_ref = avg_design("Low", t, SCENARIOS["Reference"])

    for sm in sm_grid:
        for tmax in tmax_grid:

            vals = {
                "RfSpr": rf_ref,
                "SMSpr": float(sm),
                "TMaxSpr": float(tmax)
            }

            x_high = avg_design("High", t, vals)
            x_low = avg_design("Low", t, vals)

            dvec = (x_high - x_high_ref) - (x_low - x_low_ref)
            e, se, lo, hi = est(dvec)

            vulnerability_rows.append([
                t, rf_ref, sm, tmax,
                e, se, lo, hi
            ])


vulnerability = pd.DataFrame(
    vulnerability_rows,
    columns=[
        "Time", "Rainfall", "SoilMoisture", "Tmax",
        "High_minus_Low_Sensitivity", "SE", "CI_Low", "CI_High"
    ]
)

vulnerability.to_csv(
    c.CLIMATE_DIR / "Climate_Vulnerability_Surface.csv",
    index=False
)


# ============================================================
# F. Individual climatic efect.. 
# Each variable changed independently to P25 or P75; while the other two variables remain at P50.
# ============================================================

variable_info = {
    "Rainfall": "RfSpr",
    "Soil moisture": "SMSpr",
    "Maximum temperature": "TMaxSpr"
}

individual_rows = []

for sev in c.SEVERITIES:
    for t in c.TIMES:

        reference = SCENARIOS["Reference"].copy()
        xref = avg_design(sev, t, reference)

        for variable_name, field in variable_info.items():
            for percentile, qvalue in [("P25", 0.25), ("P75", 0.75)]:

                vals = reference.copy()
                vals[field] = float(d[field].quantile(qvalue))

                x = avg_design(sev, t, vals)
                e, se, lo, hi = est(x - xref)

                individual_rows.append([
                    sev, t, variable_name, field,
                    percentile, vals[field],
                    e, se, lo, hi
                ])


individual = pd.DataFrame(
    individual_rows,
    columns=[
        "Severity", "Time", "Variable", "Field",
        "Percentile", "Climate_Value",
        "Delta_Recovery", "SE", "CI_Low", "CI_High"
    ]
)

individual.to_csv(
    c.CLIMATE_DIR / "Climate_Individual_Effects.csv",
    index=False
)


# ============================================================
# G. Summamary of surface ranges
# ============================================================

surface_summary = (
    surface.groupby(["Severity", "Time"])["Delta_Recovery"]
    .agg(["min", "max", "mean"])
    .reset_index()
    .rename(columns={
        "min": "Minimum_Delta",
        "max": "Maximum_Delta",
        "mean": "Mean_Delta"
    })
)

surface_summary["Range"] = (
    surface_summary["Maximum_Delta"] -
    surface_summary["Minimum_Delta"]
)

surface_summary.to_csv(
    c.CLIMATE_DIR / "Climate_Response_Surface_Summary.csv",
    index=False
)


vulnerability_summary = (
    vulnerability.groupby("Time")["High_minus_Low_Sensitivity"]
    .agg(["min", "max", "mean"])
    .reset_index()
    .rename(columns={
        "min": "Minimum_Sensitivity_Difference",
        "max": "Maximum_Sensitivity_Difference",
        "mean": "Mean_Sensitivity_Difference"
    })
)

vulnerability_summary["Range"] = (
    vulnerability_summary["Maximum_Sensitivity_Difference"] -
    vulnerability_summary["Minimum_Sensitivity_Difference"]
)

vulnerability_summary.to_csv(
    c.CLIMATE_DIR / "Climate_Vulnerability_Surface_Summary.csv",
    index=False
)


# ===
#  print output... 
# ==

print("=" * 72)
print("CLIMATE SCENARIO ANALYSIS V2 COMPLETE")
print("=" * 72)

print("\nClimate scenarios:")
for name, vals in SCENARIOS.items():
    print(
        f"{name:10s} | "
        f"Rf={vals['RfSpr']:.3f}, "
        f"SM={vals['SMSpr']:.6f}, "
        f"Tmax={vals['TMaxSpr']:.3f}"
    )

print("\nAdditional climate outputs:")
print("  Climate_Response_Surface.csv")
print("  Climate_Vulnerability_Surface.csv")
print("  Climate_Individual_Effects.csv")
print("  Climate_Response_Surface_Summary.csv")
print("  Climate_Vulnerability_Surface_Summary.csv")

print(f"\nOutputs: {c.CLIMATE_DIR}")