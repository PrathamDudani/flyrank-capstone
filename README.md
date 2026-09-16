# FlyRank ML Internship Capstone — Refresh / Content Opportunity Scoring

**Lane:** Refresh / Content Opportunity Scoring
**Deployed paper:** see `submission/paper_url.txt`

## What this is

A ranked action engine that scores content pages as `declining`, `stable`,
`recovering`, or `growing` using trailing search-performance signals, then
recommends an action (refresh, protect, expand, monitor) with a plain-language
reason code for each page. Trained and validated on the **real FlyRank
warehouse** (March 2026, ~3.6M GSC-covered daily rows across 176,738 pages).

## Headline result

A Random Forest trained on 14-day trailing features beats every baseline —
majority-class (0.194 macro-F1), a single-signal momentum heuristic (0.299),
and a hybrid model+heuristic policy that had won on earlier synthetic testing
(0.452) — scoring **0.633 macro-F1** on a time-holdout set and 0.632 on a
stricter unseen-page holdout. The hybrid-vs-model-alone reversal between
synthetic and real data is itself a documented finding — see the paper's
Results section.

## Repo layout — two pipelines

**Real-data pipeline** (produced the results in the paper):
```
extract_real_data.py        — DuckDB extraction/join against the gated HF warehouse
features_labels_real.py     — vectorized daily-grain features & labels, coverage filters
model_real.py                — Random Forest vs. baselines vs. hybrid, both validation regimes
real_charts_extra.py         — confusion matrix + feature importance (needs trained model)
real_action_engine.py        — final ranked output using the winning policy (model alone)
```
Requires a personal Hugging Face read token for `FlyRank/internship-warehouse`
(gated, request access via the dataset card, then generate a token under
Settings → Access Tokens). Run in Colab with the token in `userdata`.

**Synthetic fallback pipeline** (credential-free, used during early development
to validate the modeling approach before real warehouse access was available):
```
work/
  01_eda.ipynb                     — explore a synthetic panel matching the warehouse schema
  02_feature_engineering.ipynb     — time-aware, leakage-checked features & labels
  03_modeling_validation.ipynb     — Random Forest vs. two baselines, two validation regimes
  04_capstone_action_engine.ipynb  — full synthetic pipeline + ranked action engine
gen_data.py, features_labels.py, model.py, action_engine.py, charts.py, hybrid_eval.py
```

**Submission:**
```
submission/
  paper_url.txt                    — the deployed paper's URL (fill in after deploying)
paper.html                         — the deployed research paper
paper_assets/                      — chart images used in the paper
```

## Reproducing

Real pipeline (needs HF token, run in Colab):
```
python extract_real_data.py
python features_labels_real.py
python model_real.py
python real_charts_extra.py
python real_action_engine.py
```

Synthetic fallback (no credentials needed):
```
pip install -r requirements.txt
python gen_data.py
python features_labels.py
python model.py
python hybrid_eval.py
python charts.py
python action_engine.py
```

## Deploying the paper

`paper.html` is a single self-contained file that references images in
`paper_assets/`. Push this repo to GitHub, enable GitHub Pages (Settings →
Pages → deploy from branch, root), and put the resulting URL in
`submission/paper_url.txt`.

## Data credit

Built on the FlyRank ML Internship dataset — see
[flyrank.ai](https://flyrank.ai).
