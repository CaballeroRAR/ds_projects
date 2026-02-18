-- This script mirrors the Python etl.py logic for cleaning and normalization, see file in: https://github.com/CaballeroRAR/ds_projects_collabs/tree/main/1-cluster_retail_uci/src
-- ##########################
-- Industry Standard: The Medallion Architecture
-- In a professional cloud environment, we follow the Medallion Architecture:
-- Bronze (Raw): The raw_retail_... tables.
-- Silver (Cleaned/Staging): Filtered, deduplicated, and typed data. This is our etl.sql.
-- Gold (Business/Curated): Final RFM metrics and model inputs. This will be our rfm_aggregation.sql.
-- ##########################

-- SILVER TABLES
-- 1. Create Table for 2009-2010
CREATE OR REPLACE TABLE `retail_segmentation.etl_silver_2009_2010` AS
SELECT DISTINCT -- Removes duplicated rows (mirroring drop_duplicates)
  Invoice AS invoice,
  StockCode AS stock_code,
  Description AS description,
  CAST(Quantity AS INT64) AS quantity,
  InvoiceDate AS invoice_date,
  CAST(Price AS FLOAT64) AS price,
  CAST(`Customer ID` AS INT64) AS customer_id,
  Country AS country,
  (Quantity * Price) AS revenue
FROM
  `retail_segmentation.raw_retail_2009_2010`
WHERE
  -- Drop NA Customer IDs
  `Customer ID` IS NOT NULL
  -- Price and Quantity > 0
  AND Quantity > 0
  AND Price > 0
  -- Regex: Exactly 6 digits for Invoices (removes cancellations and abnormal codes)
  AND REGEXP_CONTAINS(Invoice, r'^\d{6}$')
  -- Regex: Valid StockCodes (5 digits OR 5 digits + letters)
  AND REGEXP_CONTAINS(StockCode, r'^\d{5}$|^\d{5}[a-zA-Z]+$');

-- 2. Create Table for 2010-2011 (Scoring)
CREATE OR REPLACE TABLE `retail_segmentation.etl_silver_2010_2011` AS
SELECT DISTINCT
  Invoice AS invoice,
  StockCode AS stock_code,
  Description AS description,
  CAST(Quantity AS INT64) AS quantity,
  InvoiceDate AS invoice_date,
  CAST(Price AS FLOAT64) AS price,
  CAST(`Customer ID` AS INT64) AS customer_id,
  Country AS country,
  (Quantity * Price) AS revenue
FROM
  `retail_segmentation.raw_retail_2010_2011`
WHERE
  `Customer ID` IS NOT NULL
  AND Quantity > 0
  AND Price > 0
  AND REGEXP_CONTAINS(Invoice, r'^\d{6}$')
  AND REGEXP_CONTAINS(StockCode, r'^\d{5}$|^\d{5}[a-zA-Z]+$');