-- Segment Drift Analysis
-- This query compares the distribution of business segments between 2009 and 2011.

WITH drift_data AS (
  SELECT segment_label, '2009-2010' as period FROM `retail_segmentation.final_scored_2009_2010`
  UNION ALL
  SELECT segment_label, '2010-2011' as period FROM `retail_segmentation.final_scored_2010_2011`
),
counts AS (
  SELECT
    segment_label,
    COUNTIF(period = '2009-2010') as count_2009,
    COUNTIF(period = '2010-2011') as count_2010
  FROM
    drift_data
  GROUP BY
    segment_label
)
SELECT
  segment_label,
  count_2009,
  count_2010,
  (count_2010 - count_2009) as absolute_diff,
  ROUND(SAFE_DIVIDE(count_2010 - count_2009, count_2009) * 100, 2) as pct_change
FROM
  counts
ORDER BY
  count_2010 DESC;
