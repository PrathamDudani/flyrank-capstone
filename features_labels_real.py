"""
features_labels_real.py (v2) — vectorized rewrite.

The original per-page Python loop (`for pid, g in df.groupby("page_id"): ...`)
calls a function once per page PER CUTOFF. With a large number of unique
content_hash_id values, that's potentially millions of slow Python-level
calls -- not frozen, just extremely slow. This version computes each cutoff
in a handful of vectorized groupby().agg() passes instead: 6 cutoffs means
roughly 6 x (a few groupby calls), each processing the whole table at once.
Should finish in seconds to low minutes instead of hours.
"""
import numpy as np, pandas as pd
import time

t0 = time.time()
df = pd.read_parquet("flyrank_real_daily.parquet")
df["report_date"] = pd.to_datetime(df["report_date"])

MIN_DATE = df["report_date"].min()
df["day_num"] = (df["report_date"] - MIN_DATE).dt.days + 1
print(f"Loaded {len(df):,} rows, {df.page_id.nunique():,} unique pages, "
      f"day range 1-{df.day_num.max()} ({time.time()-t0:.1f}s)")

FEATURE_WINDOW = 14
LABEL_WINDOW = 7
CUTOFFS = list(range(14, 25, 2))  # [14, 16, 18, 20, 22, 24]

MIN_FEATURE_DAYS = 7
MIN_LABEL_DAYS = 3

def expected_ctr(p):
    return 0.32 / (1 + p) ** 0.9

STATIC_COLS = ["content_type", "word_count", "char_count", "n_keywords_mapped",
               "top_search_volume", "avg_competition", "backlinks", "main_intent", "client_hash_id"]
STATIC_COLS = [c for c in STATIC_COLS if c in df.columns]

all_samples = []

for cutoff in CUTOFFS:
    t1 = time.time()
    hist = df[(df.day_num > cutoff - FEATURE_WINDOW) & (df.day_num <= cutoff)]
    fut  = df[(df.day_num > cutoff) & (df.day_num <= cutoff + LABEL_WINDOW)]

    last_third  = hist[hist.day_num > cutoff - FEATURE_WINDOW / 3]
    first_third = hist[hist.day_num <= cutoff - 2 * FEATURE_WINDOW / 3]

    # -- vectorized per-page aggregates for this cutoff --
    hist_stats = hist.groupby("page_id").agg(
        n_feature_days=("day_num", "count"),
        hist_mean_clicks=("clicks", "mean"),
        hist_std_clicks=("clicks", "std"),
    )
    last_agg = last_third.groupby("page_id").agg(
        recent_avg_clicks=("clicks", "mean"),
        recent_avg_impressions=("impressions", "mean"),
        recent_avg_position=("avg_position", "mean"),
        recent_avg_ctr=("ctr", "mean"),
    )
    first_agg = first_third.groupby("page_id").agg(
        early_avg_clicks=("clicks", "mean"),
        early_avg_pos=("avg_position", "mean"),
    )
    fut_agg = fut.groupby("page_id").agg(
        n_label_days=("day_num", "count"),
        future_avg_clicks=("clicks", "mean"),
    )
    ga4_agg = None
    if "engagement_signal" in hist.columns:
        ga4_agg = hist.groupby("page_id")["engagement_signal"].apply(lambda s: s.notna().any()).rename("has_ga4_signal")

    static_agg = hist.groupby("page_id")[STATIC_COLS].first() if STATIC_COLS else None

    # -- join everything on page_id --
    merged = hist_stats.join(last_agg, how="inner").join(fut_agg, how="inner")
    if ga4_agg is not None:
        merged = merged.join(ga4_agg, how="left")
    merged = merged.join(first_agg, how="left")  # left: early window may not exist for young pages
    if static_agg is not None:
        merged = merged.join(static_agg, how="left")

    merged = merged[(merged.n_feature_days >= MIN_FEATURE_DAYS) & (merged.n_label_days >= MIN_LABEL_DAYS)].copy()
    if merged.empty:
        print(f"cutoff {cutoff}: 0 qualifying pages ({time.time()-t1:.1f}s)")
        continue

    merged["momentum_clicks"] = np.where(
        merged["early_avg_clicks"].notna() & (merged["early_avg_clicks"] > 1e-6),
        (merged["recent_avg_clicks"] - merged["early_avg_clicks"]) / merged["early_avg_clicks"].clip(lower=1e-6),
        0.0,
    )
    merged["momentum_position"] = np.where(
        merged["early_avg_pos"].notna(),
        merged["early_avg_pos"] - merged["recent_avg_position"],
        0.0,
    )
    merged["volatility_clicks"] = (merged["hist_std_clicks"] / merged["hist_mean_clicks"].clip(lower=1e-6)).fillna(0.0)
    merged["ctr_gap_vs_expected"] = merged["recent_avg_ctr"] - expected_ctr(merged["recent_avg_position"])
    merged["future_change"] = (merged["future_avg_clicks"] - merged["recent_avg_clicks"]) / merged["recent_avg_clicks"].clip(lower=1e-6)

    def label_row(r):
        if r.future_change > 0.15 and r.momentum_clicks < 0:
            return "recovering"
        elif r.future_change > 0.15:
            return "growing"
        elif r.future_change < -0.15:
            return "declining"
        return "stable"

    merged["label"] = merged.apply(label_row, axis=1)  # small df at this point, fine to apply row-wise
    merged["cutoff_day"] = cutoff
    if "has_ga4_signal" not in merged.columns:
        merged["has_ga4_signal"] = False
    merged["has_ga4_signal"] = merged["has_ga4_signal"].fillna(False)

    merged = merged.reset_index()
    all_samples.append(merged)
    print(f"cutoff {cutoff}: {len(merged):,} qualifying pages ({time.time()-t1:.1f}s)")

samp = pd.concat(all_samples, ignore_index=True) if all_samples else pd.DataFrame()
print(f"\nTotal samples: {len(samp):,}  (total time {time.time()-t0:.1f}s)")
print(samp.label.value_counts() if len(samp) else "No samples produced.")

if len(samp) == 0:
    raise SystemExit("No samples produced -- check MIN_FEATURE_DAYS/MIN_LABEL_DAYS against your data's actual coverage.")

# ---- split ----
train_cutoffs = CUTOFFS[:-2]
test_cutoffs = CUTOFFS[-2:]

rng = np.random.default_rng(7)
all_pages = samp.page_id.unique()
holdout_pages = set(rng.choice(all_pages, size=int(0.3 * len(all_pages)), replace=False))

train = samp[samp.cutoff_day.isin(train_cutoffs) & (~samp.page_id.isin(holdout_pages))]
test_time = samp[samp.cutoff_day.isin(test_cutoffs) & (~samp.page_id.isin(holdout_pages))]
test_page_time = samp[samp.cutoff_day.isin(test_cutoffs) & (samp.page_id.isin(holdout_pages))]

print(f"Train: {len(train):,} | Test time-holdout: {len(test_time):,} | Test page+time holdout: {len(test_page_time):,}")
assert len(set(train.page_id) & set(test_page_time.page_id)) == 0, "page leakage!"

train.to_parquet("real_train.parquet", index=False)
test_time.to_parquet("real_test_time.parquet", index=False)
test_page_time.to_parquet("real_test_page_time.parquet", index=False)
samp.to_parquet("real_all_samples.parquet", index=False)

print("\nNOTE: this is a within-March split only. June 2026 is the real sealed")
print("test -- if you get access to score against it later, that result is the")
print("one that actually matters for the paper's generalization claim.")
