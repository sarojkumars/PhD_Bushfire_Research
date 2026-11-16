import numpy as np,pandas as pd,matplotlib.pyplot as plt
from appendix_config import *
FIG_DIR.mkdir(exist_ok=True)
m=pd.read_csv(VAL_DIR/"12_main_model_comparison.csv")
d=m.dropna(subset=["auc"]); plt.figure(figsize=(7,4.5)); plt.bar(d.model,d.auc); plt.ylabel("Test AUC"); plt.xlabel("Model"); plt.title("Held-out predictive discrimination"); plt.tight_layout(); plt.savefig(FIG_DIR/"Fig_model_comparison_AUC.png",dpi=400,bbox_inches="tight"); plt.close()
e=pd.read_csv(VAL_DIR/"10_environmental_domain_ablation_metrics.csv"); order=["M","T","FV","M+T","M+FV","T+FV","M+T+FV"]; e=e.set_index("domain_set").reindex(order).reset_index()
plt.figure(figsize=(8,4.5)); plt.plot(e.domain_set,e.auc,marker="o")
for x,y in zip(e.domain_set,e.auc): plt.text(x,y+.003,f"{y:.3f}",ha="center")
plt.ylabel("Test AUC"); plt.xlabel("Environmental predictor domain"); plt.title("Standalone and integrated environmental predictive information"); plt.tight_layout(); plt.savefig(FIG_DIR/"Fig_environmental_domain_integration.png",dpi=400,bbox_inches="tight"); plt.close()
cf=VAL_DIR/"07_context_specific_test_metrics_RECALCULATED.csv"
if cf.exists():
 c=pd.read_csv(cf)
 for typ in ["landform","vegetation"]:
  q=c[(c.context_type==typ)&(c.variant.isin(["CF","FC"]))]; p=q.pivot(index="context_level",columns="variant",values="auc"); ax=p.plot(kind="bar",figsize=(8,4.5)); ax.set_ylabel("Test AUC"); ax.set_xlabel(typ.capitalize()); ax.set_title(f"Predictive discrimination across {typ} contexts"); plt.tight_layout(); plt.savefig(FIG_DIR/f"Fig_context_AUC_{typ}.png",dpi=400,bbox_inches="tight"); plt.close()
bf=VAL_DIR/"13_paired_bootstrap_differences.csv"
if bf.exists():
 b=pd.read_csv(bf); b=b[b.metric=="AUC"]; y=np.arange(len(b)); plt.figure(figsize=(8,4.5)); plt.errorbar(b.difference,y,xerr=[b.difference-b.ci_low,b.ci_high-b.difference],fmt="o",capsize=4); plt.axvline(0,linewidth=1); plt.yticks(y,b.comparison); plt.xlabel("Difference in held-out AUC"); plt.title("Paired bootstrap uncertainty in model differences"); plt.tight_layout(); plt.savefig(FIG_DIR/"Fig_bootstrap_AUC_differences.png",dpi=400,bbox_inches="tight"); plt.close()
af=VAL_DIR/"08_FC_attention_weights_by_context_FINAL.csv"
if af.exists():
 a=pd.read_csv(af)
 for typ in ["landform","vegetation"]:
  p=a[a.context_type==typ].pivot(index="context_level",columns="base_model",values="mean_attention_weight"); ax=p.plot(kind="bar",figsize=(9,4.8)); ax.set_ylabel("Mean attention weight"); ax.set_xlabel(typ.capitalize()); ax.set_title(f"Classifier weighting across {typ} contexts"); plt.tight_layout(); plt.savefig(FIG_DIR/f"Fig_attention_weights_{typ}.png",dpi=400,bbox_inches="tight"); plt.close()
m.to_csv(FIG_DIR/"Table_model_comparison.csv",index=False); e.to_csv(FIG_DIR/"Table_environmental_domain_ablation.csv",index=False)
print("Saved figures/tables:",FIG_DIR)
