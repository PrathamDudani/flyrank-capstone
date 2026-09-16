import nbformat as nbf

def make_nb(cells):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {"kernelspec": {"display_name":"Python 3","language":"python","name":"python3"}}
    return nb

def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)

# ---------------- 01_eda ----------------
c1 = [
md("# Assignment 01 — Exploratory Data Analysis\n"
   "**Lane:** Refresh / Content Opportunity Scoring\n\n"
   "This notebook explores the weekly search-performance panel before any feature engineering. "
   "It's the first of the weekly assignments that built toward the capstone.\n\n"
   "**Data note:** the real capstone uses the gated FlyRank warehouse release on Hugging Face "
   "(`hf://`, read-token gated) following the workflow in starter notebook 03. For this deliverable "
   "I built a **synthetic panel that mirrors the warehouse schema** (page-level weekly impressions, "
   "clicks, avg. position, CTR, engagement, plus content metadata), because I don't have a personal "
   "HF read token for the gated release. The feature/label/model code is written to be schema-compatible: "
   "pointing `pd.read_parquet(...)` at a DuckDB export of the real `hf://` tables should work with minimal changes."),
code("import pandas as pd, numpy as np\n"
     "df = pd.read_parquet('../flyrank_synthetic_weekly.parquet')\n"
     "print(df.shape)\n"
     "df.head()"),
md("## Panel shape and coverage"),
code("print('Pages:', df.page_id.nunique())\n"
     "print('Weeks:', df.week.min(), '-', df.week.max())\n"
     "print('Content types:\\n', df.content_type.value_counts())"),
md("## Distribution of core signals"),
code("df[['impressions','clicks','avg_position','ctr','engagement_rate']].describe()"),
md("## Visual sanity check: a handful of pages over time\n"
   "Different pages clearly show different underlying trajectories (growth, decline, volatility, "
   "step-change around a refresh event) — this is the structure the modeling stage needs to pick up on."),
code("import matplotlib.pyplot as plt\n"
     "sample_pages = df.page_id.drop_duplicates().sample(6, random_state=3)\n"
     "fig, ax = plt.subplots(figsize=(9,4))\n"
     "for p in sample_pages:\n"
     "    g = df[df.page_id==p].sort_values('week')\n"
     "    ax.plot(g.week, g.clicks, label=p, alpha=0.8)\n"
     "ax.set_xlabel('week'); ax.set_ylabel('clicks'); ax.legend(fontsize=7); ax.set_title('Sample page trajectories')\n"
     "plt.tight_layout(); plt.show()"),
]

# ---------------- 02_feature_engineering ----------------
feat_src = open('/home/claude/capstone/features_labels.py').read()
c2 = [
md("# Assignment 02 — Feature Engineering & Label Design\n\n"
   "This is the core methodology notebook: turning the weekly panel into modeling samples with "
   "a **time-aware, leakage-checked** design.\n\n"
   "**Design choices:**\n"
   "- Each sample = one page scored at a `cutoff_week`, using a **trailing 12-week feature window** "
   "and a **following 6-week label window**.\n"
   "- Labels (`growing` / `declining` / `recovering` / `stable`) compare the future window average "
   "clicks to the page's own recent baseline — `recovering` specifically requires the page to have "
   "been on a *declining* trajectory earlier in the feature window before turning up.\n"
   "- Train cutoffs and test cutoffs are chosen so their feature+label windows **never overlap in time** "
   "(explicit assertion below), and a second test set holds out entire pages the model never saw in training."),
code(feat_src),
md("## Result\nThe printed output above confirms: zero time overlap between train and test windows, "
   "and zero page overlap between train and the strict page+time holdout set."),
]

# ---------------- 03_modeling_validation ----------------
model_src = open('/home/claude/capstone/model.py').read()
c3 = [
md("# Assignment 03 — Baseline, Model, and Validation\n\n"
   "Trains a Random Forest classifier on the leakage-checked samples from notebook 02, and compares "
   "it against two baselines on **two separate validation sets**:\n\n"
   "1. **Time-holdout** — same pages, strictly later time window.\n"
   "2. **Page + time holdout** — pages the model never saw in training at all, scored on a strictly "
   "later time window.\n\n"
   "Baselines:\n"
   "- **Majority-class** — always predict the most common label (`stable`).\n"
   "- **Single-signal heuristic** — the rule an analyst might apply by eye: threshold on 4-week click "
   "momentum alone."),
code(model_src),
md("## Reading the result honestly\n"
   "The model beats both baselines on macro-F1 in the time-holdout regime, but on the stricter "
   "page+time holdout, the simple heuristic is competitive on macro-F1 even though the model still wins "
   "on accuracy and on `recovering`-class recall. This is treated as a finding, not hidden — see the "
   "capstone paper's Limitations section."),
]

# ---------------- 04_capstone ----------------
action_src = open('/home/claude/capstone/action_engine.py').read()
chart_src = open('/home/claude/capstone/charts.py').read()
c4 = [
md("# Capstone Notebook — Refresh / Content Opportunity Scoring\n\n"
   "End-to-end: synthetic FlyRank-schema panel → time-aware leakage-checked features/labels → "
   "Random Forest vs two baselines, validated two ways → charts for the paper → a ranked action "
   "engine with reason codes for every currently-live page.\n\n"
   "This notebook stitches together assignments 01–03 and adds the final deliverable: the ranked "
   "action engine that the paper's recommendations section is built from."),
md("## 1. Load panel (see 01_eda.ipynb for exploration)"),
code("import pandas as pd\n"
     "df = pd.read_parquet('../flyrank_synthetic_weekly.parquet')\n"
     "df.shape"),
md("## 2. Features, labels, leakage checks (full logic in 02_feature_engineering.ipynb)"),
code("# See 02_feature_engineering.ipynb for the full, annotated version of this step.\n"
     "# Re-run here for a self-contained capstone notebook:\n" + feat_src),
md("## 3. Model + baselines + validation (full logic in 03_modeling_validation.ipynb)"),
code(model_src),
md("## 4. Charts for the paper"),
code(chart_src),
md("## 5. Ranked action engine\n"
   "Scores every page as of the most recent 12-week window, attaches predicted status, model "
   "confidence, plain-language reason codes, and a recommended action. Sorted so declining pages "
   "needing attention surface first."),
code(action_src),
md("## Output\n`ranked_action_engine.csv` is the artifact the paper's Ranked Recommendations section "
   "summarizes. `results.json` and `paper_assets/*.png` back the Results section."),
]

import os
os.makedirs('/home/claude/capstone/work', exist_ok=True)
nbf.write(make_nb(c1), '/home/claude/capstone/work/01_eda.ipynb')
nbf.write(make_nb(c2), '/home/claude/capstone/work/02_feature_engineering.ipynb')
nbf.write(make_nb(c3), '/home/claude/capstone/work/03_modeling_validation.ipynb')
nbf.write(make_nb(c4), '/home/claude/capstone/work/04_capstone_action_engine.ipynb')
print("notebooks written")
