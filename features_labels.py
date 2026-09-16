import numpy as np, pandas as pd

df = pd.read_parquet("/home/claude/capstone/flyrank_synthetic_weekly.parquet")

def expected_ctr(p):
    return 0.32 / (1 + p)**0.9

FEATURE_WINDOW = 12   # weeks of history used for features
LABEL_WINDOW = 6      # weeks ahead used to define the label
CUTOFFS_TRAIN = [14, 18, 22, 26, 30, 34]   # all history+label fully resolves by week 40
CUTOFFS_TEST  = [52, 56, 60, 64, 68, 72]   # all history starts week>=41, fully after the week-40 gap
GAP_CHECK_END_TRAIN = max(CUTOFFS_TRAIN) + LABEL_WINDOW          # 32
GAP_CHECK_START_TEST = min(CUTOFFS_TEST) - FEATURE_WINDOW + 1    # 23

def build_sample(g, cutoff):
    """g: single page's full 52-week df, sorted by week. cutoff = last week INCLUDED in feature window."""
    hist = g[(g.week > cutoff - FEATURE_WINDOW) & (g.week <= cutoff)]
    fut  = g[(g.week > cutoff) & (g.week <= cutoff + LABEL_WINDOW)]
    if len(hist) < FEATURE_WINDOW or len(fut) < LABEL_WINDOW:
        return None

    last4 = hist[hist.week > cutoff - 4]
    prior4 = hist[(hist.week > cutoff - 8) & (hist.week <= cutoff - 4)]
    first_half = hist[hist.week <= cutoff - 6]
    second_half = hist[hist.week > cutoff - 6]

    recent_avg_clicks = last4.clicks.mean()
    prior_avg_clicks = prior4.clicks.mean() if len(prior4) else np.nan
    momentum_clicks = (recent_avg_clicks - prior_avg_clicks) / max(prior_avg_clicks, 1e-6)

    recent_avg_pos = last4.avg_position.mean()
    prior_avg_pos = prior4.avg_position.mean() if len(prior4) else np.nan
    momentum_position = (prior_avg_pos - recent_avg_pos)  # positive = improved (moved up)

    early_slope = second_half.clicks.mean() - first_half.clicks.mean()  # past trend within window
    volatility_clicks = hist.clicks.std() / max(hist.clicks.mean(), 1e-6)

    exp_ctr = expected_ctr(recent_avg_pos)
    actual_ctr = last4.ctr.mean()
    ctr_gap = actual_ctr - exp_ctr

    future_avg_clicks = fut.clicks.mean()
    future_change = (future_avg_clicks - recent_avg_clicks) / max(recent_avg_clicks, 1e-6)

    if future_change > 0.15 and early_slope < 0:
        label = "recovering"
    elif future_change > 0.15:
        label = "growing"
    elif future_change < -0.15:
        label = "declining"
    else:
        label = "stable"

    row = dict(
        page_id=g.page_id.iloc[0], cutoff_week=cutoff, content_type=g.content_type.iloc[0],
        word_count=g.word_count.iloc[0], internal_links=g.internal_links.iloc[0],
        has_schema_markup=bool(g.has_schema_markup.iloc[0]),
        publish_age_days=hist.publish_age_days.iloc[-1],
        days_since_update=hist.days_since_update.iloc[-1],
        recent_avg_clicks=recent_avg_clicks, recent_avg_impressions=last4.impressions.mean(),
        recent_avg_position=recent_avg_pos, recent_avg_ctr=actual_ctr,
        momentum_clicks_4v8=momentum_clicks, momentum_position_4v8=momentum_position,
        early_within_window_slope=early_slope, volatility_clicks=volatility_clicks,
        ctr_gap_vs_expected=ctr_gap, engagement_rate=last4.engagement_rate.mean(),
        label=label, _archetype_debug=g.archetype.iloc[0],
    )
    return row

samples = []
for pid, g in df.groupby("page_id"):
    g = g.sort_values("week")
    for c in CUTOFFS_TRAIN + CUTOFFS_TEST:
        r = build_sample(g, c)
        if r: samples.append(r)

samp = pd.DataFrame(samples)
print("Total samples:", len(samp))
print(samp.label.value_counts())

# ---- leakage / window sanity checks ----
assert GAP_CHECK_END_TRAIN <= GAP_CHECK_START_TEST - 1, "train/test windows overlap in time!"
print(f"Train windows fully resolve by week {GAP_CHECK_END_TRAIN}; "
      f"earliest test feature data starts week {GAP_CHECK_START_TEST}. Gap = "
      f"{GAP_CHECK_START_TEST - GAP_CHECK_END_TRAIN} week(s). No leakage across the split.")

# grouped page holdout: 30% of pages never appear in ANY training sample
rng = np.random.default_rng(7)
all_pages = samp.page_id.unique()
holdout_pages = set(rng.choice(all_pages, size=int(0.3*len(all_pages)), replace=False))

train_mask = samp.cutoff_week.isin(CUTOFFS_TRAIN) & (~samp.page_id.isin(holdout_pages))
test_time_mask = samp.cutoff_week.isin(CUTOFFS_TEST) & (~samp.page_id.isin(holdout_pages))          # time-holdout, same pages
test_page_time_mask = samp.cutoff_week.isin(CUTOFFS_TEST) & (samp.page_id.isin(holdout_pages))       # page+time holdout, unseen pages

train = samp[train_mask].copy()
test_time = samp[test_time_mask].copy()
test_page_time = samp[test_page_time_mask].copy()

print("\nTrain:", len(train), "| Test (time-holdout, seen pages):", len(test_time),
      "| Test (page+time holdout, unseen pages):", len(test_page_time))

overlap = set(train.page_id) & set(test_page_time.page_id)
assert len(overlap) == 0, "page leakage into strict holdout!"
print("Confirmed zero page overlap between train and the page+time holdout set.")

train.to_parquet("/home/claude/capstone/train.parquet", index=False)
test_time.to_parquet("/home/claude/capstone/test_time.parquet", index=False)
test_page_time.to_parquet("/home/claude/capstone/test_page_time.parquet", index=False)
samp.to_parquet("/home/claude/capstone/all_samples.parquet", index=False)
