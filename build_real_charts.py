import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.edgecolor":"#333", "axes.labelcolor":"#222",
    "text.color":"#222", "xtick.color":"#444", "ytick.color":"#444",
    "figure.facecolor":"white", "axes.facecolor":"white"
})

# ---- REAL numbers, transcribed exactly from the executed Colab notebook ----
results = {
 "Time-holdout\n(same pages)": dict(majority=0.194, heuristic=0.299, model=0.633, hybrid=0.452),
 "Page+time holdout\n(unseen pages)": dict(majority=0.194, heuristic=0.298, model=0.632, hybrid=0.452),
}

fig, ax = plt.subplots(figsize=(7.5,4.6))
regimes = list(results.keys())
x = np.arange(len(regimes)); w = 0.2
maj = [results[r]["majority"] for r in regimes]
heur = [results[r]["heuristic"] for r in regimes]
model = [results[r]["model"] for r in regimes]
hybrid = [results[r]["hybrid"] for r in regimes]
ax.bar(x-1.5*w, maj, w, label="Majority-class baseline", color="#D9D3C7")
ax.bar(x-0.5*w, heur, w, label="Single-signal heuristic baseline", color="#B7AE9C")
ax.bar(x+0.5*w, model, w, label="Model alone (Random Forest) — final", color="#2E6B5E")
ax.bar(x+1.5*w, hybrid, w, label="Hybrid (model + heuristic override)", color="#C15B4A")
ax.set_xticks(x); ax.set_xticklabels(regimes)
ax.set_ylabel("Macro-F1 (4 classes)")
ax.set_title("Macro-F1 on the real FlyRank warehouse — March 2026")
ax.legend(frameon=False, fontsize=8.5, loc="upper right")
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout(); plt.savefig("/home/claude/real_capstone/paper_assets/fig1_real_model_vs_baseline.png", dpi=160); plt.close()

# ---- Class balance, real counts across all 669,066 page-window samples ----
labels = ["declining","stable","recovering","growing"]
counts = {"declining":123887,"stable":417404,"recovering":53983,"growing":73792}
palette = {"declining":"#C15B4A","stable":"#8C8577","recovering":"#4A8C7A","growing":"#2E6B5E"}
fig, ax = plt.subplots(figsize=(6.4,4))
ax.bar(labels, [counts[l] for l in labels], color=[palette[l] for l in labels])
ax.set_ylabel("Page-window samples")
ax.set_title("Class balance — 669,066 real page-window samples, March 2026")
ax.spines[["top","right"]].set_visible(False)
for i,l in enumerate(labels):
    ax.text(i, counts[l]+8000, f"{counts[l]:,}", ha="center", fontsize=9, color="#444")
plt.tight_layout(); plt.savefig("/home/claude/real_capstone/paper_assets/fig2_real_class_balance.png", dpi=160); plt.close()

print("Both real charts written.")
