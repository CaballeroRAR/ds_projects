-- Final Scoring Script with Business Labels
-- Based on the Interpretation of Centroids:
-- Cluster 1: High Spend, High Frequency, Low Recency -> Champions
-- Cluster 3: Mid-High Spend, Mid Frequency, Mid Recency -> Loyal
-- Cluster 2: Low-Mid Spend, Low Frequency, Low Recency -> New / Promising
-- Cluster 4: Low Spend, Low Frequency, High Recency -> Hibernating

-- 1. Create Final Scored Table for 2009-2010
CREATE OR REPLACE TABLE `retail_segmentation.final_scored_2009_2010` AS
WITH raw_predictions AS (
  SELECT
    *
  FROM
    ML.PREDICT(MODEL `retail_segmentation.customer_segments_model`,
      (SELECT * FROM `retail_segmentation.rfm_ready_2009_2010`))
)
SELECT
  *,
  CASE
    WHEN CENTROID_ID = 1 THEN 'Champions'
    WHEN CENTROID_ID = 3 THEN 'Loyal'
    WHEN CENTROID_ID = 2 THEN 'New / Promising'
    WHEN CENTROID_ID = 4 THEN 'Hibernating'
    ELSE 'Unknown'
  END AS segment_label
FROM
  raw_predictions;

-- 2. Create Final Scored Table for 2010-2011
CREATE OR REPLACE TABLE `retail_segmentation.final_scored_2010_2011` AS
WITH raw_predictions AS (
  SELECT
    *
  FROM
    ML.PREDICT(MODEL `retail_segmentation.customer_segments_model`,
      (SELECT * FROM `retail_segmentation.rfm_ready_2010_2011`))
)
SELECT
  *,
  CASE
    WHEN CENTROID_ID = 1 THEN 'Champions'
    WHEN CENTROID_ID = 3 THEN 'Loyal'
    WHEN CENTROID_ID = 2 THEN 'New / Promising'
    WHEN CENTROID_ID = 4 THEN 'Hibernating'
    ELSE 'Unknown'
  END AS segment_label
FROM
  raw_predictions;