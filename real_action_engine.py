"""
real_action_engine.py — scores every qualifying real page as of the most
recent window (cutoff day 24) using the trained model, with reason codes
built from the real feature set (no internal_links/has_schema_markup here —
those were synthetic-only; real reason codes lean on search_volume,
competition, backlinks, and momentum instead).

Run in the same Colab session, after model_real.py.
"""
import numpy as np, pandas as pd, joblib

model = joblib.load("real_refresh_opportunity_model.joblib")
samp = pd.read_parquet("real_all_samples.parquet")

NUM_FEATS = ["word_count", "char_count", "n_keywords_mapped", "top_search_volume", "avg_competition", "backlinks",
             "recent_avg_clicks", "recent_avg_impressions", "recent_avg_position", "recent_avg_ctr",
             "momentum_clicks", "momentum_position", "volatility_clicks", "ctr_gap_vs_expected"]
CAT_FEATS = ["content_type", "main_intent", "has_ga4_signal"]

cur = samp[samp.cutoff_day == samp.cutoff_day.max()].copy()  # most recent window = "now"
print(f"Scoring {len(cur):,} pages as of cutoff day {cur.cutoff_day.iloc[0]}")

for c in NUM_FEATS:
    cur[c] = pd.to_numeric(cur[c], errors="coerce").astype(float)
for c in CAT_FEATS:
    cur[c] = cur[c].astype(object).where(cur[c].notna(), "missing").astype(str)
cur[NUM_FEATS] = cur[NUM_FEATS].fillna(cur[NUM_FEATS].median())

X = cur[NUM_FEATS + CAT_FEATS]
proba = model.predict_proba(X)
classes = model.named_steps["clf"].classes_
pred = model.predict(X)
cur["predicted_status"] = pred  # model alone -- confirmed to beat the hybrid override on real data
for i, c in enumerate(classes):
    cur[f"p_{c}"] = proba[:, i]
cur["confidence"] = proba.max(axis=1)

def reason_codes(row, k=3):
    reasons = []
    if row.momentum_clicks < -0.15:
        reasons.append(f"click momentum {row.momentum_clicks*100:.0f}% vs early window")
    elif row.momentum_clicks > 0.15:
        reasons.append(f"click momentum +{row.momentum_clicks*100:.0f}% vs early window")
    if row.momentum_position > 1.5:
        reasons.append(f"avg position improved {row.momentum_position:.1f} spots")
    elif row.momentum_position < -1.5:
        reasons.append(f"avg position slipped {abs(row.momentum_position):.1f} spots")
    if row.ctr_gap_vs_expected < -0.02:
        reasons.append(f"CTR trails position-expected by {abs(row.ctr_gap_vs_expected)*100:.1f}pp")
    if row.top_search_volume and row.top_search_volume > cur.top_search_volume.quantile(0.75):
        reasons.append("targets a high-search-volume keyword")
    if row.backlinks == 0:
        reasons.append("zero backlinks")
    if not reasons:
        reasons.append("within normal range on tracked signals")
    return "; ".join(reasons[:k])

cur["reason_codes"] = cur.apply(reason_codes, axis=1)

def action_for(row):
    s = row.predicted_status
    if s == "declining":
        return "Refresh / rewrite" if row.backlinks == 0 else "Review for cannibalization or SERP change"
    if s == "recovering":
        return "Protect & monitor"
    if s == "growing":
        return "Expand — target adjacent keywords, build backlinks"
    return "Monitor" if row.ctr_gap_vs_expected < -0.02 else "Protect"

cur["recommended_action"] = cur.apply(action_for, axis=1)

priority = {"declining": 0, "recovering": 1, "growing": 2, "stable": 3}
cur["priority_rank"] = cur.predicted_status.map(priority)
cur = cur.sort_values(["priority_rank", "confidence"], ascending=[True, False]).reset_index(drop=True)
cur.insert(0, "rank", range(1, len(cur) + 1))

out_cols = ["rank", "page_id", "content_type", "predicted_status", "confidence", "recommended_action",
            "reason_codes", "recent_avg_clicks", "recent_avg_position", "momentum_clicks"]
cur[out_cols].to_csv("real_ranked_action_engine.csv", index=False)
print(cur.predicted_status.value_counts())
print("\nTop 10:")
print(cur[out_cols].head(10).to_string(index=False))
print("\nSend me real_ranked_action_engine.csv (or just the printed top 10 and the value_counts above)")
print("and I'll finish the paper's Recommendations section with real page examples.")
