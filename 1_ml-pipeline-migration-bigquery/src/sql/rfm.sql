-- This script mirrors the compute_rfm_features logic from feature_engineering.py
-- Optimized with CROSS JOIN for performance.

-- 1. GOLD TABLE: RFM for Year 1 (Training: 2009-2010)
CREATE OR REPLACE TABLE `retail_segmentation.rfm_gold_2009_2010` AS
WITH max_date_ref AS (
  -- Calculate this once for the whole dataset
  SELECT MAX(CAST(invoice_date AS DATE)) as overall_max_date 
  FROM `retail_segmentation.etl_silver_2009_2010`
),
customer_aggregation AS (
  SELECT
    customer_id,
    COUNT(DISTINCT invoice) AS frequency,
    SUM(quantity * price) AS sale_value,
    MAX(CAST(invoice_date AS DATE)) AS last_invoice_date
  FROM
    `retail_segmentation.etl_silver_2009_2010`
  GROUP BY
    customer_id
)
SELECT
  a.customer_id,
  a.sale_value,
  a.frequency,
  DATE_DIFF(m.overall_max_date, a.last_invoice_date, DAY) AS recency_days,
  -- Log Transformations (matched to np.log1p)
  LOG(a.sale_value + 1) AS log_monetary,
  LOG(a.frequency + 1) AS log_frequency,
  LOG(DATE_DIFF(m.overall_max_date, a.last_invoice_date, DAY) + 1) AS log_recency
FROM
  customer_aggregation a
CROSS JOIN 
  max_date_ref m;


-- 2. GOLD TABLE: RFM for Year 2 (Scoring: 2010-2011)
CREATE OR REPLACE TABLE `retail_segmentation.rfm_gold_2010_2011` AS
WITH max_date_ref AS (
  SELECT MAX(CAST(invoice_date AS DATE)) as overall_max_date 
  FROM `retail_segmentation.etl_silver_2010_2011`
),
customer_aggregation AS (
  SELECT
    customer_id,
    COUNT(DISTINCT invoice) AS frequency,
    SUM(quantity * price) AS sale_value,
    MAX(CAST(invoice_date AS DATE)) AS last_invoice_date
  FROM
    `retail_segmentation.etl_silver_2010_2011`
  GROUP BY
    customer_id
)
SELECT
  a.customer_id,
  a.sale_value,
  a.frequency,
  DATE_DIFF(m.overall_max_date, a.last_invoice_date, DAY) AS recency_days,
  LOG(a.sale_value + 1) AS log_monetary,
  LOG(a.frequency + 1) AS log_frequency,
  LOG(DATE_DIFF(m.overall_max_date, a.last_invoice_date, DAY) + 1) AS log_recency
FROM
  customer_aggregation a
CROSS JOIN 
  max_date_ref m;