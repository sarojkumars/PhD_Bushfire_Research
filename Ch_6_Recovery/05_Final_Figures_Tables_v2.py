import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ========
# Config...
# =========
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
    fig.savefig(c.FIGURE_DIR / f"{stem}.png", dpi=c.FIG_DPI, bbox_inches="tight")
    fig.savefig(c.FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


# ================================
# FIGURE 03 — Severity × Time
# ===============================
e1 = pd.read_csv(c.RECOVERY_DIR / "E1_Severity_Time_Predictions.csv")
fig, ax = plt.subplots(figsize=(6.6, 4.5))

for sev, colour in [("Low", c.LOW_COLOUR), ("High", c.HIGH_COLOUR)]:
    g = e1[e1.Severity == sev].set_index("Time").loc[c.TIMES].reset_index()
    x = np.arange(len(c.TIMES))
    ax.plot(x, g.Estimated_Mean, marker="o", linewidth=2, color=colour, label=sev)
    ax.fill_between(x, g.CI_Low.to_numpy(), g.CI_High.to_numpy(), color=colour, alpha=0.15)

ax.set_xticks(np.arange(len(c.TIMES)), c.TIMES)
ax.set_xlabel("Recovery stage")
ax.set_ylabel("Estimated normalised NBR recovery")
ax.set_title("Severity-conditioned post-fire recovery")
ax.legend(title="Initial severity", frameon=False)
ax.spines[["top", "right"]].set_visible(False)
save(fig, "Fig03_Severity_Recovery")


# =================================================
# FIGURE 04 — Landform and vegetation conditioning
# ===================================================
e2 = pd.read_csv(c.RECOVERY_DIR / "E2_Landform_Predictions.csv")
e3 = pd.read_csv(c.RECOVERY_DIR / "E3_Vegetation_Predictions.csv")
fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), sharex=True)

# Landform
for j, sev in enumerate(c.SEVERITIES):
    ax = axes[0, j]
    for land in c.LANDFORMS:
        g = e2[(e2.Severity == sev) & (e2.Landform == land)].set_index("Time").loc[c.TIMES].reset_index()
        ax.plot(g.Time, g.Estimated_Mean, marker="o", linewidth=1.8,
                color=c.LANDFORM_COLOURS[land], label=c.LANDFORM_LABELS[land])
    ax.set_title(f"{sev} severity - landform")
    ax.set_ylabel("Estimated normalised NBR recovery")
    ax.spines[["top", "right"]].set_visible(False)

# Vegetation
for j, sev in enumerate(c.SEVERITIES):
    ax = axes[1, j]
    for veg in c.VEGETATION:
        g = e3[(e3.Severity == sev) & (e3.Vegetation == veg)].set_index("Time").loc[c.TIMES].reset_index()
        ax.plot(g.Time, g.Estimated_Mean, marker="o", linewidth=1.8, label=c.VEGETATION_LABELS[veg])
    ax.set_title(f"{sev} severity - vegetation")
    ax.set_xlabel("Recovery stage")
    ax.set_ylabel("Estimated normalised NBR recovery")
    ax.spines[["top", "right"]].set_visible(False)

axes[0, 1].legend(title="Landform", frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
axes[1, 1].legend(title="Vegetation", frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
save(fig, "Fig04_Context_Recovery")


# ======================================================
# FIGURE 05 — Integrated Landform × Vegetation heatmap
# ======================================================
e4 = pd.read_csv(c.RECOVERY_DIR / "E4_Integrated_Predictions.csv")
e4["Scenario"] = e4.Vegetation + " – " + e4.Landform
order = [f"{veg} – {land}" for veg in c.INTEGRATED_VEGETATION for land in c.LANDFORMS]
vmin, vmax = e4.Estimated_Mean.min(), e4.Estimated_Mean.max()

fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.0), sharey=True)

for ax, sev in zip(axes, c.SEVERITIES):
    p = e4[e4.Severity == sev].pivot(index="Scenario", columns="Time", values="Estimated_Mean").reindex(index=order, columns=c.TIMES)
    im = ax.imshow(p.to_numpy(), aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title(f"{sev} severity")
    ax.set_xticks(range(len(c.TIMES)), c.TIMES)
    ax.set_yticks(range(len(order)), order)

    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            val = p.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

# Colourbar outside heatmaps
fig.subplots_adjust(left=0.18, right=0.83, bottom=0.12, top=0.90, wspace=0.10)
cbar_ax = fig.add_axes([0.86, 0.20, 0.018, 0.62])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label("Estimated normalised NBR recovery", rotation=90, labelpad=12)

# Save manually — do not call save() because it uses tight_layout()
fig.savefig(c.FIGURE_DIR / "Fig05_Integrated_Context_Heatmap.png", dpi=c.FIG_DPI, bbox_inches="tight")
fig.savefig(c.FIGURE_DIR / "Fig05_Integrated_Context_Heatmap.pdf", bbox_inches="tight")
plt.close(fig)


# ===========================================
# FIGURE 06 — CHM structural conditioning
# ===========================================
chm = pd.read_csv(c.CHM_DIR / "CHM_Percentile_Predictions.csv")
gam_path = c.CHM_DIR / "CHM_GAM_Predictions.csv"

if gam_path.exists():
    gam = pd.read_csv(gam_path)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))

    # Continuous CHM GAM
    for j, sev in enumerate(c.SEVERITIES):
        ax = axes[0, j]
        for t in ["T1", "T3", "T5"]:
            g = gam[(gam.Severity == sev) & (gam.Time == t)]
            ax.plot(g.CHM_m, g.Estimated_Recovery, linewidth=1.8, label=t)
        ax.set_title(f"{sev} severity — continuous CHM")
        ax.set_xlabel("Pre-fire canopy height (m)")
        ax.set_ylabel("Estimated normalised NBR recovery")
        ax.legend(frameon=False, title="Recovery stage")
        ax.spines[["top", "right"]].set_visible(False)

    bottom = axes[1, :]

else:
    fig, bottom = plt.subplots(1, 2, figsize=(11.5, 4.3))

# P25/P50/P75 CHM scenarios
for j, sev in enumerate(c.SEVERITIES):
    ax = bottom[j]
    for level in ["P25", "P50", "P75"]:
        g = chm[(chm.Severity == sev) & (chm.CHM_Level == level)].set_index("Time").loc[c.TIMES].reset_index()
        metres = g.CHM_m.iloc[0]
        ax.plot(g.Time, g.Estimated_Recovery, marker="o", linewidth=1.8, label=f"{level} ({metres:.0f} m)")
    ax.set_title(f"{sev} severity — CHM scenarios")
    ax.set_xlabel("Recovery stage")
    ax.set_ylabel("Predicted normalised NBR recovery")
    ax.legend(frameon=False, title="Canopy condition")
    ax.spines[["top", "right"]].set_visible(False)

save(fig, "Fig06_CHM_Structural_Conditioning")


# ================================================
# FIGURE 07 — Progressive climate perturbation
# =================================================
cp = pd.read_csv(c.CLIMATE_DIR / "Climate_Progressive_Perturbation.csv")
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)

stage_colours = {"T1": "#0072B2", "T3": "#009E73", "T5": "#D55E00"}

for ax, sev in zip(axes, c.SEVERITIES):
    for t in ["T1", "T3", "T5"]:
        for pathway, ls in [("HotDry", "-"), ("CoolWet", "--")]:
            g = cp[(cp.Severity == sev) & (cp.Time == t) & (cp.Pathway == pathway)].sort_values("Progress")
            direction = 1 if pathway == "HotDry" else -1
            x = direction * g.Progress.to_numpy() * 100
            ax.plot(x, g.Delta_Recovery, marker="o", linestyle=ls, linewidth=1.8, color=stage_colours[t])

    ax.axhline(0, linewidth=0.9)
    ax.axvline(0, linewidth=0.9)
    ax.set_title(f"{sev} severity")
    ax.set_xlabel("Climate perturbation (%)\nCool-wet ← Reference → Hot-dry")
    ax.spines[["top", "right"]].set_visible(False)

axes[0].set_ylabel("Δ normalised NBR recovery")

# Separate legends: colour = recovery stage; line style = climate scenario
stage_handles = [
    Line2D([0], [0], color=stage_colours["T1"], marker="o", linewidth=1.8, label="T1"),
    Line2D([0], [0], color=stage_colours["T3"], marker="o", linewidth=1.8, label="T3"),
    Line2D([0], [0], color=stage_colours["T5"], marker="o", linewidth=1.8, label="T5")
]

scenario_handles = [
    Line2D([0], [0], color="black", linestyle="-", linewidth=2, label="Hot-dry"),
    Line2D([0], [0], color="black", linestyle="--", linewidth=2, label="Cool-wet")
]

legend_stage = axes[1].legend(handles=stage_handles, title="Recovery stage", frameon=False,
                              bbox_to_anchor=(1.02, 1.00), loc="upper left")
axes[1].add_artist(legend_stage)

axes[1].legend(handles=scenario_handles, title="Climate scenario", frameon=False,
               bbox_to_anchor=(1.02, 0.58), loc="upper left")

save(fig, "Fig07_Climate_Progressive_Scenario")


# ======================================================
# FIGURE 08 — Landform-conditioned climate sensitivity
# =======================================================
cl = pd.read_csv(c.CLIMATE_DIR / "Climate_Landform_Endpoint_Contrasts.csv")
fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), sharex=True, sharey=True)

for ax, land in zip(axes.ravel(), c.LANDFORMS):
    for sev, colour in [("Low", c.LOW_COLOUR), ("High", c.HIGH_COLOUR)]:
        for pathway, ls in [("HotDry", "-"), ("CoolWet", "--")]:
            g = cl[(cl.Landform == land) & (cl.Severity == sev) & (cl.Pathway == pathway)].set_index("Time").loc[c.TIMES].reset_index()
            ax.plot(g.Time, g.Delta_Recovery, marker="o", linewidth=1.8, linestyle=ls, color=colour)

    ax.axhline(0, linewidth=0.9)
    ax.set_title(c.LANDFORM_LABELS[land])
    ax.set_xlabel("Recovery stage")
    ax.set_ylabel("Δ normalised NBR recovery")
    ax.spines[["top", "right"]].set_visible(False)

# Separate legends: colour = severity; line style = climate scenario
severity_handles = [
    Line2D([0], [0], color=c.LOW_COLOUR, marker="o", linewidth=2, linestyle="-", label="Low"),
    Line2D([0], [0], color=c.HIGH_COLOUR, marker="o", linewidth=2, linestyle="-", label="High")
]

scenario_handles = [
    Line2D([0], [0], color="black", linestyle="-", linewidth=2, label="Hot-dry"),
    Line2D([0], [0], color="black", linestyle="--", linewidth=2, label="Cool-wet")
]

legend_severity = axes[0, 1].legend(handles=severity_handles, title="Initial severity", frameon=False,
                                   bbox_to_anchor=(1.02, 1.00), loc="upper left")
axes[0, 1].add_artist(legend_severity)

axes[0, 1].legend(handles=scenario_handles, title="Climate scenario", frameon=False,
                  bbox_to_anchor=(1.02, 0.62), loc="upper left")

save(fig, "Fig08_Climate_Landform_Sensitivity")


# =================
#  Tables 
# =============
pd.read_csv(c.RECOVERY_DIR / "Primary_Omnibus_Tests.csv").to_csv(
    c.TABLE_DIR / "Table_Key_Omnibus_Tests.csv", index=False)

pd.read_csv(c.CLIMATE_DIR / "Climate_Landform_Model_Comparison.csv").to_csv(
    c.TABLE_DIR / "Table_Climate_Landform_Model_Comparison.csv", index=False)

endpoint = cp[cp.Progress == 1.0].copy()
endpoint[["Severity", "Time", "Pathway", "Delta_Recovery", "Delta_CI_Low", "Delta_CI_High"]].to_csv(
    c.TABLE_DIR / "Table_Climate_Endpoint_Summary.csv", index=False)


# ================
# Print.. 
# ================
print("=" * 72)
print("FINAL FIGURES + TABLES V2 COMPLETE")
print("=" * 72)
print(f"Figures: {c.FIGURE_DIR}")
print(f"Tables : {c.TABLE_DIR}")