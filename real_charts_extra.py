"""
real_charts_extra.py — run this in the SAME Colab session right after
model_real.py (or in a fresh session after re-loading the saved artifacts).
Produces the two charts fig1/fig2 already have real counterparts for, plus
the ones that need the actual trained model: confusion matrix and feature
importance.
"""
import numpy as np, pandas as pd, joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.edgecolor": "#333", "axes.labelcolor": "#222",
    "text.color": "#222", "xtick.color": "#444", "ytick.color": "#444",
    "figure.facecolor": "white", "axes.facecolor": "white"
})
LABELS = ["declining", "stable", "recovering", "growing"]

model = joblib.load("real_refresh_opportunity_model.joblib")
test_time = pd.read_parquet("real_test_time.parquet")

NUM_FEATS = ["word_count", "char_count", "n_keywords_mapped", "top_search_volume", "avg_competition", "backlinks",
             "recent_avg_clicks", "recent_avg_impressions", "recent_avg_position", "recent_avg_ctr",
             "momentum_clicks", "momentum_position", "volatility_clicks", "ctr_gap_vs_expected"]
CAT_FEATS = ["content_type", "main_intent", "has_ga4_signal"]

for c in NUM_FEATS:
    test_time[c] = pd.to_numeric(test_time[c], errors="coerce").astype(float)
for c in CAT_FEATS:
    test_time[c] = test_time[c].astype(object).where(test_time[c].notna(), "missing").astype(str)
test_time[NUM_FEATS] = test_time[NUM_FEATS].fillna(test_time[NUM_FEATS].median())

pred = model.predict(test_time[NUM_FEATS + CAT_FEATS])
cm = confusion_matrix(test_time["label"], pred, labels=LABELS)
cm_norm = cm / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(6.4, 5.0))
ax.imshow(cm_norm, cmap="Greens", vmin=0, vmax=1)
ax.set_xticks(range(4)); ax.set_xticklabels(LABELS, rotation=30, ha="right")
ax.set_yticks(range(4)); ax.set_yticklabels(LABELS)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title("Confusion matrix — model alone\ntime-holdout set, real data (row-normalized)", fontsize=13)
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center",
                 color="white" if cm_norm[i,j] > 0.5 else "#222", fontsize=9)
plt.tight_layout(); plt.savefig("fig3_real_confusion_matrix.png", dpi=160); plt.close()
print("Per-class recall (diagonal):", dict(zip(LABELS, np.diag(cm_norm).round(3))))

imp = model.named_steps["clf"].feature_importances_
feat_names = NUM_FEATS + list(model.named_steps["pre"].named_transformers_["cat"].get_feature_names_out(CAT_FEATS))
imp_df = pd.DataFrame({"feature": feat_names, "importance": imp}).sort_values("importance", ascending=False).head(10).iloc[::-1]

fig, ax = plt.subplots(figsize=(7, 4.6))
ax.barh(imp_df.feature, imp_df.importance, color="#4A8C7A")
ax.set_xlabel("Random Forest importance")
ax.set_title("Which real signals drive the model's predictions")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig("fig4_real_feature_importance.png", dpi=160); plt.close()

print("\nDownload fig3_real_confusion_matrix.png and fig4_real_feature_importance.png")
print("and send them back along with the printed per-class recall numbers above.")
