import pandas as pd, numpy as np, joblib, json
from sklearn.metrics import f1_score, accuracy_score, classification_report

model = joblib.load("refresh_opportunity_model.joblib")
NUM_FEATS = ["word_count","internal_links","publish_age_days","days_since_update",
             "recent_avg_clicks","recent_avg_impressions","recent_avg_position","recent_avg_ctr",
             "momentum_clicks_4v8","momentum_position_4v8","early_within_window_slope",
             "volatility_clicks","ctr_gap_vs_expected","engagement_rate"]
CAT_FEATS = ["content_type","has_schema_markup"]
LABELS = ["declining","stable","recovering","growing"]

def hybrid_predict(d):
    base = model.predict(d[NUM_FEATS+CAT_FEATS])
    override = d["momentum_clicks_4v8"] < -0.15
    out = base.copy()
    out[override.values] = "declining"
    return out

results = {}
for name, path in [("test_time (same pages, future window)","test_time.parquet"),
                    ("test_page_time (unseen pages, future window)","test_page_time.parquet")]:
    d = pd.read_parquet(path)
    y = d["label"]
    pred_model = model.predict(d[NUM_FEATS+CAT_FEATS])
    pred_hybrid = hybrid_predict(d)
    results[name] = dict(
        model_macro_f1=f1_score(y,pred_model,average='macro',labels=LABELS,zero_division=0),
        model_acc=accuracy_score(y,pred_model),
        hybrid_macro_f1=f1_score(y,pred_hybrid,average='macro',labels=LABELS,zero_division=0),
        hybrid_acc=accuracy_score(y,pred_hybrid),
        hybrid_report=classification_report(y,pred_hybrid,labels=LABELS,zero_division=0,output_dict=True),
    )
    print(name)
    print(" model    macroF1=%.3f acc=%.3f" % (results[name]['model_macro_f1'], results[name]['model_acc']))
    print(" hybrid   macroF1=%.3f acc=%.3f" % (results[name]['hybrid_macro_f1'], results[name]['hybrid_acc']))
    print(" hybrid declining recall=%.3f" % results[name]['hybrid_report']['declining']['recall'])

json.dump(results, open("hybrid_results.json","w"), indent=2, default=float)
