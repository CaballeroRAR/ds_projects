-- This script applies Log transformations to raw RFM metrics for ML.
-- Tables: rfm_ready_2009_2010, rfm_ready_2010_2011

-- 1. READY TABLE: RFM for Year 1 (Training: 2009-2010)
CREATE OR REPLACE TABLE `retail_segmentation.rfm_ready_2009_2010` AS
SELECT
  customer_id,
  sale_value,
  frequency,
  recency_days,
  -- Log Transformations (matched to np.log1p)
  LOG(sale_value + 1) AS log_monetary,
  LOG(frequency + 1) AS log_frequency,
  LOG(recency_days + 1) AS log_recency
FROM
  `retail_segmentation.rfm_2009_2010`;


-- 2. READY TABLE: RFM for Year 2 (Scoring: 2010-2011)
CREATE OR REPLACE TABLE `retail_segmentation.rfm_ready_2010_2011` AS
SELECT
  customer_id,
  sale_value,
  frequency,
  recency_days,
  LOG(sale_value + 1) AS log_monetary,
  LOG(frequency + 1) AS log_frequency,
  LOG(recency_days + 1) AS log_recency
FROM
  `retail_segmentation.rfm_2010_2011`;
