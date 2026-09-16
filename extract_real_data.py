# Run this as a SINGLE, SELF-CONTAINED cell in Colab. It does not depend on
# any earlier cell's connection or views still being alive — every DuckDB
# session in Colab is independent unless you're reusing the exact same `con`
# object, so this recreates everything from scratch each time.

!pip install -q duckdb --upgrade
import duckdb
from google.colab import userdata
hf_token = userdata.get('HF_TOKEN')
con = duckdb.connect()
con.sql("INSTALL httpfs; LOAD httpfs;")
con.sql(f"""
CREATE OR REPLACE SECRET hf_token (TYPE huggingface, TOKEN '{hf_token}');
""")

# ---- 0. Recreate the source views in THIS connection ----
con.sql("""
CREATE OR REPLACE VIEW daily_perf AS
SELECT * FROM 'hf://datasets/FlyRank/internship-warehouse/fact_content_daily_performance/month=2026-03/data_0.parquet'
""")
con.sql("""
CREATE OR REPLACE VIEW dim_clients AS
SELECT * FROM 'hf://datasets/FlyRank/internship-warehouse/dim_clients.parquet'
""")
con.sql("""
CREATE OR REPLACE VIEW dim_content AS
SELECT * FROM 'hf://datasets/FlyRank/internship-warehouse/dim_content.parquet'
""")

# ---- 1. STOP AND LOOK: check dim_content's real columns BEFORE building
#         the join below. ----
con.sql("DESCRIBE dim_content").show()
con.sql("SELECT * FROM dim_content LIMIT 5").show()

# ---- 2. IMPORTANT: dim_content is at (client, content, keyword, url) grain,
#         not (client, content) grain -- a content piece can target several
#         keywords, so it appears multiple times. Collapse it to ONE row per
#         content_hash_id first, or the join below will silently duplicate
#         daily_perf rows for every extra keyword match. ----
con.sql("""
CREATE OR REPLACE VIEW content_static AS
SELECT
    content_hash_id,
    ANY_VALUE(content_type)         AS content_type,
    ANY_VALUE(word_count)           AS word_count,
    ANY_VALUE(char_count)           AS char_count,
    ANY_VALUE(content_created_date) AS publish_date,
    ANY_VALUE(content_updated_date) AS last_updated_date,
    ANY_VALUE(is_published)         AS is_published,
    COUNT(DISTINCT keyword_hash_id) AS n_keywords_mapped,
    MAX(search_volume)              AS top_search_volume,
    AVG(competition)                AS avg_competition,
    MAX(cpc)                        AS max_cpc,
    MAX(backlinks)                  AS backlinks,
    ANY_VALUE(main_intent)          AS main_intent
FROM dim_content
WHERE is_deleted IS NOT TRUE
GROUP BY content_hash_id
""")

# ---- 3. Now join fact table to the one-row-per-page content_static view ----
con.sql("""
CREATE OR REPLACE VIEW daily_perf_enriched AS
SELECT
    d.report_date,
    d.client_hash_id,
    d.content_hash_id,
    d.gsc_data_available,
    d.ga4_data_available,
    d.gsc_impressions,
    d.gsc_clicks,
    d.gsc_avg_position,
    d.ga4_engaged_sessions,
    d.scroll_events,
    c.content_type,
    c.word_count,
    c.char_count,
    c.publish_date,
    c.last_updated_date,
    c.is_published,
    c.n_keywords_mapped,
    c.top_search_volume,
    c.avg_competition,
    c.max_cpc,
    c.backlinks,
    c.main_intent
FROM daily_perf d
LEFT JOIN content_static c ON d.content_hash_id = c.content_hash_id
WHERE d.gsc_data_available IS TRUE               -- drop rows with no GSC data at all
  AND d.gsc_impressions >= 1                     -- avoid divide-by-zero noise
""")

# ---- 4. Aggregate daily rows up to page-week (content_hash_id x week) ----
# NOTE: with only ~31 days of March, this gives ~4-5 week buckets per page --
# too few for the original 12-week feature / 6-week label window design.
# See the note below about redesigning cutoffs for daily grain instead.
weekly = con.sql("""
SELECT
    content_hash_id                                  AS page_id,
    client_hash_id,
    DATE_TRUNC('week', report_date)                  AS week_start,
    SUM(gsc_impressions)                             AS impressions,
    SUM(gsc_clicks)                                  AS clicks,
    AVG(gsc_avg_position)                            AS avg_position,
    SUM(gsc_clicks) * 1.0 / NULLIF(SUM(gsc_impressions), 0) AS ctr,
    AVG(ga4_engaged_sessions)                        AS engagement_rate,  -- will be mostly NULL, see notes
    ANY_VALUE(content_type)                          AS content_type,
    ANY_VALUE(word_count)                            AS word_count,
    ANY_VALUE(char_count)                            AS char_count,
    ANY_VALUE(publish_date)                          AS publish_date,
    ANY_VALUE(last_updated_date)                     AS last_updated_date,
    ANY_VALUE(is_published)                          AS is_published,
    ANY_VALUE(n_keywords_mapped)                     AS n_keywords_mapped,
    ANY_VALUE(top_search_volume)                     AS top_search_volume,
    ANY_VALUE(avg_competition)                       AS avg_competition,
    ANY_VALUE(max_cpc)                               AS max_cpc,
    ANY_VALUE(backlinks)                             AS backlinks,
    ANY_VALUE(main_intent)                           AS main_intent
FROM daily_perf_enriched
GROUP BY content_hash_id, client_hash_id, DATE_TRUNC('week', report_date)
ORDER BY page_id, week_start
""").df()

print(weekly.shape)
print(weekly.head())
weekly.to_parquet("flyrank_real_weekly.parquet", index=False)

# ---- Also save a DAILY version, which is what you actually want to model on ----
daily = con.sql("""
SELECT
    content_hash_id      AS page_id,
    client_hash_id,
    report_date,
    gsc_impressions       AS impressions,
    gsc_clicks             AS clicks,
    gsc_avg_position       AS avg_position,
    gsc_clicks * 1.0 / NULLIF(gsc_impressions, 0) AS ctr,
    ga4_engaged_sessions   AS engagement_signal,
    content_type, word_count, char_count, publish_date, last_updated_date,
    is_published, n_keywords_mapped, top_search_volume, avg_competition,
    max_cpc, backlinks, main_intent
FROM daily_perf_enriched
""").df()

print(daily.shape)
daily.to_parquet("flyrank_real_daily.parquet", index=False)
