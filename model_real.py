import numpy as np, pandas as pd, json, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, f1_score, confusion_matrix, accuracy_score

train = pd.read_parquet("real_train.parquet")
test_time = pd.read_parquet("real_test_time.parquet")
test_page_time = pd.read_parquet("real_test_page_time.parquet")

NUM_FEATS = ["word_count", "char_count", "n_keywords_mapped", "top_search_volume", "avg_competition", "backlinks",
             "recent_avg_clicks", "recent_avg_impressions", "recent_avg_position", "recent_avg_ctr",
             "momentum_clicks", "momentum_position", "volatility_clicks", "ctr_gap_vs_expected"]
CAT_FEATS = ["content_type", "main_intent", "has_ga4_signal"]
LABELS = ["declining", "stable", "recovering", "growing"]

# ---- clean nulls / dtypes before sklearn sees them ----
# Real data has genuine missing values (e.g. some dim_content rows lack
# word_count). Pandas' nullable Int64 dtype represents those as pd.NA, which
# scikit-learn's ColumnTransformer can't handle -- coerce to plain float64
# (uses np.nan instead) and impute with the TRAIN set's median, applied
# consistently to both test sets (no leakage: medians come from train only).
for d in (train, test_time, test_page_time):
    for c in NUM_FEATS:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce").astype(float)
    for c in CAT_FEATS:
        if c in d.columns:
            d[c] = d[c].astype(object)
            d[c] = d[c].where(d[c].notna(), "missing").astype(str)

train_medians = train[NUM_FEATS].median()
for d in (train, test_time, test_page_time):
    d[NUM_FEATS] = d[NUM_FEATS].fillna(train_medians)

# ---- cap training set size for reasonable training time ----
# Real data may have far more unique pages than a synthetic test run. A few
# hundred thousand rows is already plenty for a Random Forest at this depth;
# training on millions of rows single-threaded can run for a very long time.
MAX_TRAIN_ROWS = 300_000
if len(train) > MAX_TRAIN_ROWS:
    print(f"Training set has {len(train):,} rows -- sampling down to {MAX_TRAIN_ROWS:,} for training time.")
    train = train.sample(n=MAX_TRAIN_ROWS, random_state=42)

def Xy(d):
    return d[NUM_FEATS + CAT_FEATS], d["label"]

Xtr, ytr = Xy(train)
Xte1, yte1 = Xy(test_time)
Xte2, yte2 = Xy(test_page_time)

pre = ColumnTransformer([
    ("num", "passthrough", NUM_FEATS),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATS),
])

model = Pipeline([
    ("pre", pre),
    ("clf", RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=3,
                                    class_weight="balanced", random_state=42, n_jobs=-1))
])
print(f"Training on {len(Xtr):,} rows...")
t_fit = __import__("time").time()
model.fit(Xtr, ytr)
print(f"Trained in {__import__('time').time() - t_fit:.1f}s")

maj_class = ytr.value_counts().idxmax()
def majority_pred(y): return np.array([maj_class] * len(y))

def heuristic_pred(d):
    out = []
    for m in d["momentum_clicks"]:
        if m > 0.15: out.append("growing")
        elif m < -0.15: out.append("declining")
        else: out.append("stable")
    return np.array(out)

def hybrid_pred(d):
    base = model.predict(d[NUM_FEATS + CAT_FEATS])
    override = d["momentum_clicks"] < -0.15
    out = base.copy()
    out[override.values] = "declining"
    return out

results = {}
for name, X, y, d in [("test_time", Xte1, yte1, test_time), ("test_page_time", Xte2, yte2, test_page_time)]:
    pm, pmaj, ph, phy = model.predict(X), majority_pred(y), heuristic_pred(d), hybrid_pred(d)
    results[name] = dict(
        model_macro_f1=f1_score(y, pm, average="macro", labels=LABELS, zero_division=0),
        model_acc=accuracy_score(y, pm),
        majority_macro_f1=f1_score(y, pmaj, average="macro", labels=LABELS, zero_division=0),
        heuristic_macro_f1=f1_score(y, ph, average="macro", labels=LABELS, zero_division=0),
        hybrid_macro_f1=f1_score(y, phy, average="macro", labels=LABELS, zero_division=0),
        hybrid_acc=accuracy_score(y, phy),
        hybrid_report=classification_report(y, phy, labels=LABELS, zero_division=0, output_dict=True),
    )
    print(name, {k: round(v, 3) if isinstance(v, float) else "…" for k, v in results[name].items()})

json.dump(results, open("real_results.json", "w"), indent=2, default=float)
joblib.dump(model, "real_refresh_opportunity_model.joblib")

imp = model.named_steps["clf"].feature_importances_
feat_names = NUM_FEATS + list(model.named_steps["pre"].named_transformers_["cat"].get_feature_names_out(CAT_FEATS))
pd.DataFrame({"feature": feat_names, "importance": imp}).sort_values("importance", ascending=False)\
    .to_csv("real_feature_importance.csv", index=False)

print("\nSample sizes are much smaller than the synthetic run and the month is short —")
print("expect noisier metrics. Compare the SHAPE of the result (hybrid > model > heuristic")
print("> majority, or wherever it lands) rather than treating exact numbers as final.")
