import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

spec = importlib.util.spec_from_file_location("config", Path(__file__).with_name("00_Config_v2.py"))
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
df = pd.read_csv(c.ANALYSIS_READY)

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 9})

def save(fig, stem):
    fig.tight_layout(); fig.savefig(c.SUPP_RECOVERY_DIR / f"{stem}.png", dpi=c.FIG_DPI, bbox_inches="tight"); fig.savefig(c.SUPP_RECOVERY_DIR / f"{stem}.pdf", bbox_inches="tight"); plt.close(fig)

# S01 sample distribution
unique = df.drop_duplicates("Sample_ID")
sample = unique.groupby(["Severity", "Landform", "Vegetation"], dropna=False).size().rename("N").reset_index()
sample.to_csv(c.SUPP_RECOVERY_DIR / "S01_Sample_Distribution.csv", index=False)

# S02 direct Low–High contrasts
contrast = pd.read_csv(c.RECOVERY_DIR / "E1_Low_High_Contrasts.csv")
fig, ax = plt.subplots(figsize=(6.4, 4.0)); x = np.arange(len(c.TIMES))
ax.errorbar(x, contrast.Low_minus_High, yerr=[contrast.Low_minus_High - contrast.CI_Low, contrast.CI_High - contrast.Low_minus_High], marker="o", capsize=3)
ax.axhline(0, linewidth=0.9); ax.set_xticks(x, c.TIMES); ax.set_xlabel("Recovery stage"); ax.set_ylabel("Low − High normalised NBR recovery"); ax.set_title("Severity contrast through recovery time"); ax.spines[["top", "right"]].set_visible(False)
save(fig, "S02_Low_High_Severity_Contrasts")

# S03 sustained TTR80 and TTR100
wide = df.pivot(index="Sample_ID", columns="Time", values="Recovery").reindex(columns=c.TIMES)
meta = df.drop_duplicates("Sample_ID").set_index("Sample_ID")[["Severity", "Landform", "Vegetation"]]
wide = wide.join(meta, how="left")
def sustained_ttr(row, threshold):
    vals = [row.get(t, np.nan) for t in c.TIMES]
    for i in range(len(vals)):
        rem = vals[i:]
        if any(pd.isna(v) for v in rem): continue
        if all(v >= threshold for v in rem): return c.TIMES[i]
    return "Not reached"
tables = []
for threshold in c.RECOVERY_THRESHOLDS:
    label = f"TTR{int(threshold * 100)}"; wide[label] = wide.apply(sustained_ttr, axis=1, threshold=threshold)
    for grouping in [["Severity"], ["Severity", "Landform"], ["Severity", "Vegetation"]]:
        tmp = wide.reset_index().groupby(grouping + [label], dropna=False).size().rename("N").reset_index()
        tmp["Percent"] = 100 * tmp.N / tmp.groupby(grouping).N.transform("sum"); tmp["Threshold"] = threshold; tmp["Grouping"] = " × ".join(grouping); tables.append(tmp)
pd.concat(tables, ignore_index=True).to_csv(c.SUPP_RECOVERY_DIR / "S03_Sustained_TTR80_TTR100.csv", index=False)

# S04 landform spread
e2 = pd.read_csv(c.RECOVERY_DIR / "E2_Landform_Predictions.csv")
spread = e2.groupby(["Severity", "Time"]).Estimated_Mean.agg(["min", "max"]).reset_index(); spread["Range"] = spread["max"] - spread["min"]
fig, ax = plt.subplots(figsize=(6.4, 4.0))
for sev, colour in [("Low", c.LOW_COLOUR), ("High", c.HIGH_COLOUR)]:
    g = spread[spread.Severity == sev].set_index("Time").loc[c.TIMES].reset_index(); ax.plot(g.Time, g.Range, marker="o", linewidth=1.8, color=colour, label=sev)
ax.set_xlabel("Recovery stage"); ax.set_ylabel("Landform recovery range"); ax.set_title("Contextual separation among landforms"); ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
save(fig, "S04_Landform_Context_Range")

# S05 integrated ranking/divergence
e4 = pd.read_csv(c.RECOVERY_DIR / "E4_Integrated_Predictions.csv"); e4["Scenario"] = e4.Vegetation + " – " + e4.Landform
e4["Recovery_Rank"] = e4.groupby(["Severity", "Time"]).Estimated_Mean.rank(method="min", ascending=False).astype(int)
e4.to_csv(c.SUPP_RECOVERY_DIR / "S05_Integrated_Scenario_Ranking.csv", index=False)
div = e4.groupby(["Severity", "Time"]).Estimated_Mean.agg(Minimum="min", Maximum="max", Mean="mean").reset_index(); div["Range"] = div.Maximum - div.Minimum
div.to_csv(c.SUPP_RECOVERY_DIR / "S05_Integrated_Scenario_Divergence.csv", index=False)


print("SUPPLEMENTARY RECOVERY OUTPUTS COMPLETE")
print(c.SUPP_RECOVERY_DIR)
