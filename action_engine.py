import pandas as pd, numpy as np, joblib

model = joblib.load("refresh_opportunity_model.joblib")
df = pd.read_parquet("flyrank_synthetic_weekly.parquet")

NUM_FEATS = ["word_count","internal_links","publish_age_days","days_since_update",
             "recent_avg_clicks","recent_avg_impressions","recent_avg_position","recent_avg_ctr",
             "momentum_clicks_4v8","momentum_position_4v8","early_within_window_slope",
             "volatility_clicks","ctr_gap_vs_expected","engagement_rate"]
CAT_FEATS = ["content_type","has_schema_markup"]

def expected_ctr(p): return 0.32/(1+p)**0.9

CUTOFF = 84  # score every page as of "now" using the most recent 12 weeks, no future needed
rows = []
for pid, g in df.groupby("page_id"):
    g = g.sort_values("week")
    hist = g[(g.week > CUTOFF-12) & (g.week <= CUTOFF)]
    if len(hist) < 12: continue
    last4 = hist[hist.week > CUTOFF-4]; prior4 = hist[(hist.week>CUTOFF-8)&(hist.week<=CUTOFF-4)]
    first_half = hist[hist.week<=CUTOFF-6]; second_half = hist[hist.week>CUTOFF-6]
    recent_avg_clicks = last4.clicks.mean(); prior_avg_clicks = prior4.clicks.mean()
    momentum_clicks = (recent_avg_clicks-prior_avg_clicks)/max(prior_avg_clicks,1e-6)
    recent_avg_pos = last4.avg_position.mean(); prior_avg_pos = prior4.avg_position.mean()
    momentum_position = prior_avg_pos - recent_avg_pos
    early_slope = second_half.clicks.mean() - first_half.clicks.mean()
    volatility = hist.clicks.std()/max(hist.clicks.mean(),1e-6)
    exp_ctr = expected_ctr(recent_avg_pos); actual_ctr = last4.ctr.mean(); ctr_gap = actual_ctr-exp_ctr
    rows.append(dict(page_id=pid, content_type=g.content_type.iloc[0], word_count=g.word_count.iloc[0],
        internal_links=g.internal_links.iloc[0], has_schema_markup=bool(g.has_schema_markup.iloc[0]),
        publish_age_days=hist.publish_age_days.iloc[-1], days_since_update=hist.days_since_update.iloc[-1],
        recent_avg_clicks=recent_avg_clicks, recent_avg_impressions=last4.impressions.mean(),
        recent_avg_position=recent_avg_pos, recent_avg_ctr=actual_ctr,
        momentum_clicks_4v8=momentum_clicks, momentum_position_4v8=momentum_position,
        early_within_window_slope=early_slope, volatility_clicks=volatility,
        ctr_gap_vs_expected=ctr_gap, engagement_rate=last4.engagement_rate.mean()))

cur = pd.DataFrame(rows)
X = cur[NUM_FEATS+CAT_FEATS]
proba = model.predict_proba(X)
classes = model.named_steps["clf"].classes_
pred = model.predict(X)
# Hybrid policy (validated in hybrid_eval.py): model prediction, overridden to "declining"
# whenever the simple momentum heuristic fires -- this measurably improves macro-F1 and
# declining-class recall on both held-out validation sets vs. the model alone.
override = cur["momentum_clicks_4v8"] < -0.15
pred_hybrid = pred.copy()
pred_hybrid[override.values] = "declining"
cur["predicted_status"] = pred_hybrid
cur["model_only_status"] = pred
for i,c in enumerate(classes):
    cur[f"p_{c}"] = proba[:,i]
cur["confidence"] = proba.max(axis=1)

# population medians for reason-code context
med = cur[NUM_FEATS].median()

def reason_codes(row, k=3):
    reasons = []
    if row.momentum_clicks_4v8 < -0.15:
        reasons.append(f"click momentum -{abs(row.momentum_clicks_4v8)*100:.0f}% vs prior 4wk")
    elif row.momentum_clicks_4v8 > 0.15:
        reasons.append(f"click momentum +{row.momentum_clicks_4v8*100:.0f}% vs prior 4wk")
    if row.momentum_position_4v8 > 1.5:
        reasons.append(f"avg position improved {row.momentum_position_4v8:.1f} spots")
    elif row.momentum_position_4v8 < -1.5:
        reasons.append(f"avg position slipped {abs(row.momentum_position_4v8):.1f} spots")
    if row.ctr_gap_vs_expected < -med_ctr_gap_std:
        reasons.append(f"CTR trails position-expected by {abs(row.ctr_gap_vs_expected)*100:.1f}pp — metadata review candidate")
    elif row.ctr_gap_vs_expected > med_ctr_gap_std:
        reasons.append(f"CTR beats position-expected by {row.ctr_gap_vs_expected*100:.1f}pp")
    if row.days_since_update > 365:
        reasons.append(f"not updated in {int(row.days_since_update)}d")
    if row.volatility_clicks > cur.volatility_clicks.quantile(0.8):
        reasons.append("high week-to-week volatility")
    if row.early_within_window_slope < -med_slope_std and row.momentum_clicks_4v8 > 0:
        reasons.append("was declining, now turning up")
    if not reasons:
        reasons.append("within normal range on tracked signals")
    return "; ".join(reasons[:k])

med_ctr_gap_std = cur.ctr_gap_vs_expected.std()*0.5
med_slope_std = cur.early_within_window_slope.std()*0.5
cur["reason_codes"] = cur.apply(reason_codes, axis=1)

def action_for(row):
    s = row.predicted_status
    if s == "declining":
        return "Refresh / rewrite" if row.days_since_update>300 else "Review for cannibalization or SERP change"
    if s == "recovering":
        return "Protect & monitor — reinforce what's working"
    if s == "growing":
        return "Expand — add internal links / related content"
    return "Monitor" if row.ctr_gap_vs_expected < -0.02 else "Protect"

cur["recommended_action"] = cur.apply(action_for, axis=1)

priority = {"declining":0,"recovering":1,"growing":2,"stable":3}
cur["priority_rank"] = cur.predicted_status.map(priority)
cur = cur.sort_values(["priority_rank","confidence"], ascending=[True,False]).reset_index(drop=True)
cur.insert(0, "rank", range(1, len(cur)+1))

out_cols = ["rank","page_id","content_type","predicted_status","confidence","recommended_action",
            "reason_codes","recent_avg_clicks","recent_avg_position","momentum_clicks_4v8","days_since_update"]
cur[out_cols].to_csv("/home/claude/capstone/ranked_action_engine.csv", index=False)
print(cur[out_cols].head(15).to_string(index=False))
print("\nStatus counts:\n", cur.predicted_status.value_counts())
