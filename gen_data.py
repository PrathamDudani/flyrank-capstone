import numpy as np, pandas as pd

rng = np.random.default_rng(42)

N_PAGES = 600
N_WEEKS = 90
content_types = ["blog", "product", "guide", "landing"]
type_probs = [0.45, 0.20, 0.25, 0.10]

pages = []
for i in range(N_PAGES):
    pid = f"page_{i:04d}"
    ctype = rng.choice(content_types, p=type_probs)
    publish_age_start = rng.integers(60, 1500)          # days old at week 1
    word_count = int(rng.normal({"blog":1400,"product":650,"guide":2200,"landing":500}[ctype], 300))
    word_count = max(150, word_count)
    internal_links = max(0, int(rng.normal(8, 4)))
    has_schema = rng.random() < 0.35
    last_update_offset = rng.integers(0, 500)  # days since update at week 1

    # latent quality/authority baseline -> sets base position
    authority = rng.normal(0, 1)
    base_position = np.clip(18 - authority*4 + rng.normal(0,2), 1, 60)

    # assign an underlying weekly-level trajectory archetype (unknown to model)
    archetype = rng.choice(
        ["steady_grow","steady_decline","volatile_recover","flat","seasonal","decay_then_refresh"],
        p=[0.15,0.15,0.15,0.30,0.10,0.15]
    )

    base_impr = np.clip(rng.lognormal(mean=6.5, sigma=1.0), 50, 40000)

    positions = np.zeros(N_WEEKS)
    impressions = np.zeros(N_WEEKS)
    pos = base_position
    for w in range(N_WEEKS):
        if archetype == "steady_grow":
            drift = -0.03
        elif archetype == "steady_decline":
            drift = 0.035
        elif archetype == "volatile_recover":
            drift = 0.04 if w < N_WEEKS*0.55 else -0.09
        elif archetype == "flat":
            drift = 0.0
        elif archetype == "seasonal":
            drift = 0.6*np.sin(2*np.pi*w/26) * 0.05
        else:  # decay_then_refresh: refreshed content signal at ~week 34
            if w == 34:
                pos = max(1, pos - rng.uniform(3,8))
            drift = 0.03 if w < 34 else -0.02

        pos = np.clip(pos + drift + rng.normal(0, 0.6), 1, 80)
        positions[w] = pos

        impr_trend = {
            "steady_grow": 0.010, "steady_decline": -0.010, "flat": 0.0,
            "volatile_recover": 0.012 if w < N_WEEKS*0.55 else -0.02,
            "seasonal": 0.5*np.sin(2*np.pi*w/26)*0.03,
            "decay_then_refresh": 0.006 if w < 34 else 0.018,
        }[archetype]
        base_impr = max(20, base_impr * (1 + impr_trend + rng.normal(0, 0.05)))
        impressions[w] = base_impr

    # expected CTR curve by position (rough real-world-like decay)
    def expected_ctr(p):
        return 0.32 / (1 + p)**0.9

    ctr_noise_bias = rng.normal(0, 0.15)  # page-level CTR quality relative to position (title/meta quality)
    ctr = np.clip(expected_ctr(positions) * (1 + ctr_noise_bias) * rng.normal(1,0.08,N_WEEKS), 0.001, 0.5)
    clicks = np.maximum(0, np.round(impressions * ctr))
    engagement = np.clip(0.55 - 0.004*positions + rng.normal(0,0.05,N_WEEKS) + (0.05 if has_schema else 0), 0.05, 0.95)

    for w in range(N_WEEKS):
        pages.append(dict(
            page_id=pid, content_type=ctype, week=w+1,
            publish_age_days=publish_age_start + w*7,
            days_since_update=last_update_offset + w*7 if not (archetype=="decay_then_refresh" and w>=34) else (w-34)*7,
            word_count=word_count, internal_links=internal_links, has_schema_markup=has_schema,
            impressions=round(impressions[w],1), avg_position=round(positions[w],2),
            clicks=int(clicks[w]), ctr=round(clicks[w]/max(impressions[w],1),4),
            engagement_rate=round(engagement[w],3), archetype=archetype
        ))

df = pd.DataFrame(pages)
df.to_parquet("/home/claude/capstone/flyrank_synthetic_weekly.parquet", index=False)
print(df.shape)
print(df.head(3).to_string())
print(df.groupby("archetype")["page_id"].nunique())
