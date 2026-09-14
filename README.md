# FlyRank ML Internship Capstone — Refresh / Content Opportunity Scoring

**Lane:** Refresh / Content Opportunity Scoring
**Deployed paper:** see `submission/paper_url.txt`

## What this is

A ranked action engine that scores content pages as `declining`, `stable`,
`recovering`, or `growing` using trailing search-performance signals, then
recommends an action (refresh, protect, expand, monitor) with a plain-language
reason code for each page.

## Data note (read this first)

The FlyRank warehouse on Hugging Face is gated behind a personal read token.
This project doesn't have one, so it runs on a **synthetic weekly panel built
to match the warehouse's schema** (page-level impressions, clicks, avg.
position, CTR, engagement, plus content metadata, across ~90 weeks and 600
pages, with six built-in trend archetypes). Every claim in the paper is a
claim about this synthetic panel, not about real FlyRank client data. The
feature/label/model code is written to be schema-compatible with the real
`hf://` warehouse via the starter notebook 03 workflow (DuckDB aggregation →
sklearn) — swapping the data source should require minimal changes.

## Repo layout

```
work/
  01_eda.ipynb                     — explore the panel
  02_feature_engineering.ipynb     — time-aware, leakage-checked features & labels
  03_modeling_validation.ipynb     — Random Forest vs. two baselines, two validation regimes
  04_capstone_action_engine.ipynb  — full pipeline + ranked action engine (the capstone notebook)
submission/
  paper_url.txt                    — the deployed paper's URL (fill in after deploying)
paper_assets/                      — chart images used in the paper
gen_data.py, features_labels.py, model.py, action_engine.py, charts.py, hybrid_eval.py
  — the source scripts the notebooks are built from
flyrank_synthetic_weekly.parquet   — the synthetic panel
ranked_action_engine.csv           — the final ranked output
results.json / hybrid_results.json — validation metrics
paper.html                         — the deployed research paper (open directly, or host via GitHub Pages)
```

## Reproducing

```
pip install -r requirements.txt
python gen_data.py            # synthetic panel
python features_labels.py     # features, labels, leakage checks
python model.py                # baseline + model + validation
python hybrid_eval.py         # hybrid policy validation
python charts.py              # figures
python action_engine.py       # final ranked output
```

## Deploying the paper

`paper.html` is a single self-contained file. Push this repo to GitHub, enable
GitHub Pages (Settings → Pages → deploy from branch, root), and put the
resulting URL in `submission/paper_url.txt`. Any static host works equally
well — the file has no server dependency.

## Data credit

Built on the FlyRank ML Internship dataset structure — see
[flyrank.ai](https://flyrank.ai).
