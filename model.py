import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import json

train = pd.read_parquet("/home/claude/capstone/train.parquet")
test_time = pd.read_parquet("/home/claude/capstone/test_time.parquet")
test_page_time = pd.read_parquet("/home/claude/capstone/test_page_time.parquet")

NUM_FEATS = ["word_count","internal_links","publish_age_days","days_since_update",
             "recent_avg_clicks","recent_avg_impressions","recent_avg_position","recent_avg_ctr",
             "momentum_clicks_4v8","momentum_position_4v8","early_within_window_slope",
             "volatility_clicks","ctr_gap_vs_expected","engagement_rate"]
CAT_FEATS = ["content_type","has_schema_markup"]
LABELS = ["declining","stable","recovering","growing"]

def Xy(d):
    return d[NUM_FEATS+CAT_FEATS], d["label"]

Xtr, ytr = Xy(train)
Xte1, yte1 = Xy(test_time)
Xte2, yte2 = Xy(test_page_time)

pre = ColumnTransformer([
    ("num","passthrough", NUM_FEATS),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATS),
])

model = Pipeline([
    ("pre", pre),
    ("clf", RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_leaf=3,
                                    class_weight="balanced", random_state=42))
])
model.fit(Xtr, ytr)

# ---- Baseline 1: majority class ----
maj_class = ytr.value_counts().idxmax()
def majority_pred(y): return np.array([maj_class]*len(y))

# ---- Baseline 2: naive single-signal heuristic an analyst might use by eye ----
def heuristic_pred(d):
    out = []
    for m in d["momentum_clicks_4v8"]:
        if m > 0.15: out.append("growing")
        elif m < -0.15: out.append("declining")
        else: out.append("stable")
    return np.array(out)

results = {}
for name, X, y in [("test_time (same pages, future window)", Xte1, yte1),
                    ("test_page_time (unseen pages, future window)", Xte2, yte2)]:
    pred_model = model.predict(X)
    pred_maj = majority_pred(y)
    pred_heur = heuristic_pred(X)
    results[name] = {
        "model_macro_f1": f1_score(y, pred_model, average="macro", labels=LABELS, zero_division=0),
        "model_accuracy": (pred_model==y.values).mean(),
        "majority_baseline_macro_f1": f1_score(y, pred_maj, average="macro", labels=LABELS, zero_division=0),
        "majority_baseline_accuracy": (pred_maj==y.values).mean(),
        "heuristic_baseline_macro_f1": f1_score(y, pred_heur, average="macro", labels=LABELS, zero_division=0),
        "heuristic_baseline_accuracy": (pred_heur==y.values).mean(),
        "model_report": classification_report(y, pred_model, labels=LABELS, zero_division=0, output_dict=True),
        "confusion_matrix": confusion_matrix(y, pred_model, labels=LABELS).tolist(),
    }

print(json.dumps({k: {kk:vv for kk,vv in v.items() if kk not in ("model_report","confusion_matrix")}
                   for k,v in results.items()}, indent=2))

with open("/home/claude/capstone/results.json","w") as f:
    json.dump(results, f, indent=2)

# feature importances
importances = model.named_steps["clf"].feature_importances_
feat_names = NUM_FEATS + list(model.named_steps["pre"].named_transformers_["cat"].get_feature_names_out(CAT_FEATS))
imp_df = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values("importance", ascending=False)
imp_df.to_csv("/home/claude/capstone/feature_importance.csv", index=False)
print("\nTop features:\n", imp_df.head(10).to_string(index=False))

import joblib
joblib.dump(model, "/home/claude/capstone/refresh_opportunity_model.joblib")
