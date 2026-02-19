-- This script aggregates raw RFM metrics from Silver tables for DQA.
-- Tables: rfm_2009_2010, rfm_2010_2011

-- 1. RFM for Year 1 (Training: 2009-2010)
CREATE OR REPLACE TABLE `retail_segmentation.rfm_2009_2010` AS
WITH max_date_ref AS (
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
  DATE_DIFF(m.overall_max_date, a.last_invoice_date, DAY) AS recency_days
FROM
  customer_aggregation a
CROSS JOIN 
  max_date_ref m;


-- 2. RFM for Year 2 (Scoring: 2010-2011)
CREATE OR REPLACE TABLE `retail_segmentation.rfm_2010_2011` AS
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
  DATE_DIFF(m.overall_max_date, a.last_invoice_date, DAY) AS recency_days
FROM
  customer_aggregation a
CROSS JOIN 
  max_date_ref m;
