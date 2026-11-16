import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

spec = importlib.util.spec_from_file_location("config", Path(__file__).with_name("00_Config_v2.py"))
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
})


def save(fig, stem):
    fig.tight_layout()
    fig.savefig(c.SUPP_CLIMATE_DIR / f"{stem}.png", dpi=c.FIG_DPI, bbox_inches="tight")
    fig.savefig(c.SUPP_CLIMATE_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


# ============================
# S01 — climate distribution
# =============================
df = pd.read_csv(c.ANALYSIS_READY)
d = df[df["Vegetation"] == c.PRIMARY_VEGETATION].dropna(
    subset=["RfSpr", "SMSpr", "TMaxSpr", "Sample_ID"]
).copy()

climate_fields = {
    "RfSpr": "Spring rainfall",
    "SMSpr": "Spring soil moisture",
    "TMaxSpr": "Spring maximum temperature"
}

fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))

for ax, (field, label) in zip(axes, climate_fields.items()):
    ax.hist(d[field], bins=40, edgecolor="white", linewidth=0.5)
    ax.axvline(d[field].quantile(0.25), linestyle="--", linewidth=1.0)
    ax.axvline(d[field].quantile(0.50), linestyle="-", linewidth=1.2)
    ax.axvline(d[field].quantile(0.75), linestyle="--", linewidth=1.0)
    ax.set_title(label)
    ax.set_xlabel(label)
    ax.set_ylabel("Frequency")
    ax.spines[["top", "right"]].set_visible(False)

save(fig, "S01_Climate_Distributions")


# ==================================
# S02 — Climate correlation matrix..
# ==================================
corr = d[["RfSpr", "SMSpr", "TMaxSpr"]].corr()

fig, ax = plt.subplots(figsize=(5.2, 4.5))
im = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="RdBu_r")

labels = ["Rainfall", "Soil moisture", "Tmax"]
ax.set_xticks(range(3), labels, rotation=35, ha="right")
ax.set_yticks(range(3), labels)

for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")

fig.subplots_adjust(right=0.82)
cbar_ax = fig.add_axes([0.85, 0.18, 0.025, 0.64])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label("Pearson correlation", rotation=90, labelpad=10)

fig.savefig(c.SUPP_CLIMATE_DIR / "S02_Climate_Correlation.png", dpi=c.FIG_DPI, bbox_inches="tight")
fig.savefig(c.SUPP_CLIMATE_DIR / "S02_Climate_Correlation.pdf", bbox_inches="tight")
plt.close(fig)


# ============================================================
# S03 — climate scenario..
# ============================================================
scenario = pd.read_csv(c.CLIMATE_DIR / "Climate_Scenario_Values.csv", index_col=0)
scenario.index.name = "Scenario"
scenario.reset_index().to_csv(c.SUPP_CLIMATE_DIR / "S03_Climate_Scenario_Definitions.csv", index=False)


# ============================================================
# S04 — Progressive climatic perturbation.. All T1–T5
# ============================================================
cp = pd.read_csv(c.CLIMATE_DIR / "Climate_Progressive_Perturbation.csv")

stage_colours = {
    "T1": "#0072B2",
    "T2": "#56B4E9",
    "T3": "#009E73",
    "T4": "#E69F00",
    "T5": "#D55E00"
}

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), sharey=True)

for ax, sev in zip(axes, c.SEVERITIES):
    for t in c.TIMES:
        for pathway, ls in [("HotDry", "-"), ("CoolWet", "--")]:
            g = cp[(cp.Severity == sev) & (cp.Time == t) & (cp.Pathway == pathway)].sort_values("Progress")
            direction = 1 if pathway == "HotDry" else -1
            x = direction * g.Progress.to_numpy() * 100

            ax.plot(x, g.Delta_Recovery, marker="o", linewidth=1.6,
                    linestyle=ls, color=stage_colours[t])

    ax.axhline(0, linewidth=0.8)
    ax.axvline(0, linewidth=0.8)
    ax.set_title(f"{sev} severity")
    ax.set_xlabel("Climate perturbation (%)\nCool-wet ← Reference → Hot-dry")
    ax.spines[["top", "right"]].set_visible(False)

axes[0].set_ylabel("Δ normalised NBR recovery")

stage_handles = [
    Line2D([0], [0], color=stage_colours[t], marker="o", linewidth=1.8, label=t)
    for t in c.TIMES
]

path_handles = [
    Line2D([0], [0], color="black", linestyle="-", linewidth=2, label="Hot-dry"),
    Line2D([0], [0], color="black", linestyle="--", linewidth=2, label="Cool-wet")
]

leg1 = axes[1].legend(handles=stage_handles, title="Recovery stage", frameon=False,
                      bbox_to_anchor=(1.02, 1.00), loc="upper left")
axes[1].add_artist(leg1)

axes[1].legend(handles=path_handles, title="Climate scenario", frameon=False,
               bbox_to_anchor=(1.02, 0.42), loc="upper left")

save(fig, "S04_Progressive_Climate_All_Stages")


# ============================================================
# S05 — SEVERITY-DEPENDENT climate sensitivity.. 
# ============================================================
sens = pd.read_csv(c.CLIMATE_DIR / "Climate_Severity_Sensitivity_Difference.csv")

fig, ax = plt.subplots(figsize=(7.0, 4.5))

for pathway, ls in [("HotDry", "-"), ("CoolWet", "--")]:
    g = sens[sens.Pathway == pathway].set_index("Time").loc[c.TIMES].reset_index()

    ax.errorbar(
        g.Time,
        g.High_minus_Low_Sensitivity,
        yerr=[
            g.High_minus_Low_Sensitivity - g.CI_Low,
            g.CI_High - g.High_minus_Low_Sensitivity
        ],
        marker="o",
        linestyle=ls,
        linewidth=1.8,
        capsize=3,
        label="Hot-dry" if pathway == "HotDry" else "Cool-wet"
    )

ax.axhline(0, linewidth=0.9)
ax.set_xlabel("Recovery stage")
ax.set_ylabel("High − Low climate sensitivity")
ax.set_title("Severity-dependent climate sensitivity")
ax.legend(frameon=False, title="Climate scenario")
ax.spines[["top", "right"]].set_visible(False)

save(fig, "S05_Severity_Climate_Sensitivity")


# ============================================================
# S06 — LANDFORM-CONDITIONED climate endpoint heatmap.. 
# ============================================================
land = pd.read_csv(c.CLIMATE_DIR / "Climate_Landform_Endpoint_Contrasts.csv")
land["Scenario"] = land["Landform"] + " – " + land["Severity"]

row_order = [f"{landform} – {sev}" for landform in c.LANDFORMS for sev in c.SEVERITIES]
limit = np.nanmax(np.abs(land["Delta_Recovery"]))

fig, axes = plt.subplots(1, 2, figsize=(10.5, 6.0), sharey=True)

for ax, pathway in zip(axes, ["HotDry", "CoolWet"]):
    g = land[land.Pathway == pathway]
    p = g.pivot(index="Scenario", columns="Time", values="Delta_Recovery").reindex(
        index=row_order, columns=c.TIMES
    )

    im = ax.imshow(p.to_numpy(), aspect="auto", vmin=-limit, vmax=limit, cmap="RdBu_r")
    ax.set_title("Hot-dry" if pathway == "HotDry" else "Cool-wet")
    ax.set_xticks(range(len(c.TIMES)), c.TIMES)
    ax.set_yticks(range(len(row_order)), row_order)
    ax.set_xlabel("Recovery stage")

    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            val = p.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)

fig.subplots_adjust(left=0.20, right=0.86, bottom=0.12, top=0.90, wspace=0.10)
cbar_ax = fig.add_axes([0.89, 0.20, 0.018, 0.62])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label("Δ normalised NBR recovery", rotation=90, labelpad=12)

fig.savefig(c.SUPP_CLIMATE_DIR / "S06_Landform_Climate_Heatmap.png", dpi=c.FIG_DPI, bbox_inches="tight")
fig.savefig(c.SUPP_CLIMATE_DIR / "S06_Landform_Climate_Heatmap.pdf", bbox_inches="tight")
plt.close(fig)


# ============================================================
# S07 — JOINT TMAX × SOIL-MOISTURE RESPONSE SURFACE
# Rainfall fixed at P50
# ============================================================
surface = pd.read_csv(c.CLIMATE_DIR / "Climate_Response_Surface.csv")
surface_times = ["T1", "T3", "T5"]

limit = np.nanmax(np.abs(surface["Delta_Recovery"]))
levels = np.linspace(-limit, limit, 21)

fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.5), sharex=True, sharey=True)

for i, sev in enumerate(c.SEVERITIES):
    for j, t in enumerate(surface_times):
        ax = axes[i, j]
        g = surface[(surface.Severity == sev) & (surface.Time == t)]

        p = g.pivot(index="SoilMoisture", columns="Tmax", values="Delta_Recovery")
        X, Y = np.meshgrid(p.columns.to_numpy(), p.index.to_numpy())

        cf = ax.contourf(X, Y, p.to_numpy(), levels=levels, cmap="RdBu_r", extend="both")
        ax.contour(X, Y, p.to_numpy(), levels=[0], colors="black", linewidths=1.0)

        ax.set_title(f"{sev} severity — {t}")

        if i == 1:
            ax.set_xlabel("Spring maximum temperature (°C)")

        if j == 0:
            ax.set_ylabel("Spring soil moisture")

fig.subplots_adjust(left=0.08, right=0.88, bottom=0.10, top=0.92, wspace=0.15, hspace=0.22)
cbar_ax = fig.add_axes([0.90, 0.18, 0.018, 0.64])
cbar = fig.colorbar(cf, cax=cbar_ax)
cbar.set_label("Δ normalised NBR recovery", rotation=90, labelpad=12)

fig.savefig(c.SUPP_CLIMATE_DIR / "S07_Climate_Response_Surface.png", dpi=c.FIG_DPI, bbox_inches="tight")
fig.savefig(c.SUPP_CLIMATE_DIR / "S07_Climate_Response_Surface.pdf", bbox_inches="tight")
plt.close(fig)


# ===============================================
# S08 — HIGH–LOW climate vulnerability surface.. 
# ================================================
vuln = pd.read_csv(c.CLIMATE_DIR / "Climate_Vulnerability_Surface.csv")

limit = np.nanmax(np.abs(vuln["High_minus_Low_Sensitivity"]))
levels = np.linspace(-limit, limit, 21)

fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), sharex=True, sharey=True)

for ax, t in zip(axes, surface_times):
    g = vuln[vuln.Time == t]

    p = g.pivot(
        index="SoilMoisture",
        columns="Tmax",
        values="High_minus_Low_Sensitivity"
    )

    X, Y = np.meshgrid(p.columns.to_numpy(), p.index.to_numpy())

    cf = ax.contourf(X, Y, p.to_numpy(), levels=levels, cmap="RdBu_r", extend="both")
    ax.contour(X, Y, p.to_numpy(), levels=[0], colors="black", linewidths=1.0)

    ax.set_title(t)
    ax.set_xlabel("Spring maximum temperature (°C)")
    ax.spines[["top", "right"]].set_visible(False)

axes[0].set_ylabel("Spring soil moisture")

fig.subplots_adjust(left=0.08, right=0.88, bottom=0.15, top=0.90, wspace=0.15)
cbar_ax = fig.add_axes([0.90, 0.18, 0.018, 0.64])
cbar = fig.colorbar(cf, cax=cbar_ax)
cbar.set_label("High − Low climate sensitivity", rotation=90, labelpad=12)

fig.savefig(c.SUPP_CLIMATE_DIR / "S08_Severity_Vulnerability_Surface.png", dpi=c.FIG_DPI, bbox_inches="tight")
fig.savefig(c.SUPP_CLIMATE_DIR / "S08_Severity_Vulnerability_Surface.pdf", bbox_inches="tight")
plt.close(fig)


# ===============================================
# S09 — INDIVIDUAL CLIMATE-EFFECT.. 
# ===============================================
ind = pd.read_csv(c.CLIMATE_DIR / "Climate_Individual_Effects.csv")

variables = ["Rainfall", "Soil moisture", "Maximum temperature"]

fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.0), sharex=False)

for i, sev in enumerate(c.SEVERITIES):
    for j, variable in enumerate(variables):
        ax = axes[i, j]
        g = ind[(ind.Severity == sev) & (ind.Variable == variable)].copy()

        y = np.arange(len(c.TIMES))
        offset = {"P25": -0.10, "P75": 0.10}

        for percentile, marker in [("P25", "o"), ("P75", "s")]:
            p = g[g.Percentile == percentile].set_index("Time").loc[c.TIMES].reset_index()

            ax.errorbar(
                p.Delta_Recovery,
                y + offset[percentile],
                xerr=[
                    p.Delta_Recovery - p.CI_Low,
                    p.CI_High - p.Delta_Recovery
                ],
                fmt=marker,
                capsize=3,
                linewidth=1.2,
                label=percentile
            )

        ax.axvline(0, linewidth=0.9)
        ax.set_yticks(y, c.TIMES)
        ax.set_title(f"{sev} — {variable}")
        ax.set_xlabel("Δ normalised NBR recovery")
        ax.spines[["top", "right"]].set_visible(False)

axes[0, 2].legend(title="Climate level", frameon=False,
                  bbox_to_anchor=(1.02, 1), loc="upper left")

save(fig, "S09_Individual_Climate_Effects")


# ============================================================
# S10 — RESPONSE-SURFACE summary..
# ============================================================
pd.read_csv(c.CLIMATE_DIR / "Climate_Response_Surface_Summary.csv").to_csv(
    c.SUPP_CLIMATE_DIR / "S10_Climate_Response_Surface_Summary.csv",
    index=False
)

pd.read_csv(c.CLIMATE_DIR / "Climate_Vulnerability_Surface_Summary.csv").to_csv(
    c.SUPP_CLIMATE_DIR / "S10_Climate_Vulnerability_Surface_Summary.csv",
    index=False
)


# ===========================
# S11 — climate effec ttable 
# ===========================
ind.to_csv(
    c.SUPP_CLIMATE_DIR / "S11_Individual_Climate_Effects.csv",
    index=False
)


# =================================================
# S12 — COMPLETE LANDFORM CLIMATE CONTRAST TABLE
# =================================================
land.to_csv(
    c.SUPP_CLIMATE_DIR / "S12_Landform_Climate_Endpoint_Contrasts.csv",
    index=False
)


# ========================================
# S13 — CLIMATE MODEL COMPARISON table.. 
# ========================================
comparison = pd.read_csv(c.CLIMATE_DIR / "Climate_Landform_Model_Comparison.csv")
comparison.to_csv(
    c.SUPP_CLIMATE_DIR / "S13_Climate_Landform_Model_Comparison.csv",
    index=False
)


# ===========================================
# S14 — PRIMARY CLIMATE MODEL Info.. 
# ===========================================
model_info = pd.read_csv(c.CLIMATE_DIR / "Climate_Primary_Model_Info.csv")
model_info.to_csv(
    c.SUPP_CLIMATE_DIR / "S14_Climate_Primary_Model_Info.csv",
    index=False
)


# ============================================================
# print.. 
# ============================================================
print("=" * 72)
print("SUPPLEMENTARY CLIMATE ANALYSIS V2 COMPLETE")
print("=" * 72)

print("\nSupplementary figures:")
print("  S01 Climate distributions")
print("  S02 Climate correlation")
print("  S04 Progressive perturbation — all T1–T5")
print("  S05 Severity-dependent climate sensitivity")
print("  S06 Landform climate heatmap")
print("  S07 Tmax × soil-moisture response surface")
print("  S08 High–Low climate vulnerability surface")
print("  S09 Individual climate-effect forest plots")

print("\nSupplementary tables:")
print("  S03 Climate scenario definitions")
print("  S10 Response-surface summaries")
print("  S11 Individual climate effects")
print("  S12 Landform climate endpoint contrasts")
print("  S13 Climate landform model comparison")
print("  S14 Primary climate model information")

print(f"\nOutputs: {c.SUPP_CLIMATE_DIR}")