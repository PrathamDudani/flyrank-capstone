import json, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.edgecolor":"#333", "axes.labelcolor":"#222",
    "text.color":"#222", "xtick.color":"#444", "ytick.color":"#444",
    "figure.facecolor":"white", "axes.facecolor":"white"
})

PALETTE = {"declining":"#C15B4A","stable":"#8C8577","recovering":"#4A8C7A","growing":"#2E6B5E"}
LABELS = ["declining","stable","recovering","growing"]

results = json.load(open("results.json"))
hybrid = json.load(open("hybrid_results.json"))

# 1. Model vs baselines vs hybrid, macro-F1, both validation regimes
fig, ax = plt.subplots(figsize=(7.5,4.4))
regimes = list(results.keys())
short = ["Time-holdout\n(same pages)", "Page+time holdout\n(unseen pages)"]
x = np.arange(len(regimes)); w = 0.2
model_f1 = [results[r]["model_macro_f1"] for r in regimes]
maj_f1 = [results[r]["majority_baseline_macro_f1"] for r in regimes]
heur_f1 = [results[r]["heuristic_baseline_macro_f1"] for r in regimes]
hyb_f1 = [hybrid[r]["hybrid_macro_f1"] for r in regimes]
ax.bar(x-1.5*w, maj_f1, w, label="Majority-class baseline", color="#D9D3C7")
ax.bar(x-0.5*w, heur_f1, w, label="Single-signal heuristic baseline", color="#B7AE9C")
ax.bar(x+0.5*w, model_f1, w, label="Model alone (Random Forest)", color="#6FA893")
ax.bar(x+1.5*w, hyb_f1, w, label="Hybrid (model + heuristic override) — final", color="#2E6B5E")
ax.set_xticks(x); ax.set_xticklabels(short)
ax.set_ylabel("Macro-F1 (4 classes)")
ax.set_title("Macro-F1 across two validation regimes")
ax.legend(frameon=False, fontsize=8.5, loc="upper right")
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout(); plt.savefig("/home/claude/capstone/paper_assets/fig1_model_vs_baseline.png", dpi=160); plt.close()

# 2. Confusion matrix (time-holdout) -- for the HYBRID policy, the one actually shipped
from sklearn.metrics import confusion_matrix as _cm
import joblib as _jl
_model = _jl.load("/home/claude/capstone/refresh_opportunity_model.joblib")
_NUM = ["word_count","internal_links","publish_age_days","days_since_update",
        "recent_avg_clicks","recent_avg_impressions","recent_avg_position","recent_avg_ctr",
        "momentum_clicks_4v8","momentum_position_4v8","early_within_window_slope",
        "volatility_clicks","ctr_gap_vs_expected","engagement_rate"]
_CAT = ["content_type","has_schema_markup"]
_test = pd.read_parquet("/home/claude/capstone/test_time.parquet")
_base = _model.predict(_test[_NUM+_CAT])
_override = _test["momentum_clicks_4v8"] < -0.15
_hybrid_pred = _base.copy(); _hybrid_pred[_override.values] = "declining"
cm = _cm(_test["label"], _hybrid_pred, labels=LABELS)
cm_norm = cm / cm.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(6.4,5.0))
im = ax.imshow(cm_norm, cmap="Greens", vmin=0, vmax=1)
ax.set_xticks(range(4)); ax.set_xticklabels(LABELS, rotation=30, ha="right")
ax.set_yticks(range(4)); ax.set_yticklabels(LABELS)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title("Confusion matrix — hybrid policy\ntime-holdout set (row-normalized)", fontsize=13)
for i in range(4):
    for j in range(4):
        ax.text(j,i,f"{cm_norm[i,j]:.2f}", ha="center", va="center",
                 color="white" if cm_norm[i,j]>0.5 else "#222", fontsize=9)
plt.tight_layout(); plt.savefig("/home/claude/capstone/paper_assets/fig2_confusion_matrix.png", dpi=160); plt.close()

# 3. Feature importance
imp = pd.read_csv("feature_importance.csv").head(10).iloc[::-1]
fig, ax = plt.subplots(figsize=(7,4.6))
ax.barh(imp.feature, imp.importance, color="#4A8C7A")
ax.set_xlabel("Random Forest importance")
ax.set_title("Which signals drive the model's predictions")
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout(); plt.savefig("/home/claude/capstone/paper_assets/fig3_feature_importance.png", dpi=160); plt.close()

# 4. Label distribution
samp = pd.read_parquet("all_samples.parquet")
counts = samp.label.value_counts().reindex(LABELS)
fig, ax = plt.subplots(figsize=(6.4,4))
ax.bar(counts.index, counts.values, color=[PALETTE[l] for l in counts.index])
ax.set_ylabel("Page-window samples")
ax.set_title("Class balance across all 7,200 page-window samples")
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout(); plt.savefig("/home/claude/capstone/paper_assets/fig4_class_balance.png", dpi=160); plt.close()

print("charts written")
