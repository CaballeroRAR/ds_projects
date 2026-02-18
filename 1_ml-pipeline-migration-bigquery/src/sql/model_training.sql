-- This script trains a K-means clustering model using the 2009-2010 RFM data.
CREATE OR REPLACE MODEL `retail_segmentation.customer_segments_model`
OPTIONS(
  MODEL_TYPE='KMEANS',
  NUM_CLUSTERS=4,           -- Based in elbow method from https://github.com/CaballeroRAR/ds_projects_collabs/tree/main/1-cluster_retail_uci/src
  KMEANS_INIT_METHOD='KMEANS++',
  STANDARDIZE_FEATURES=TRUE -- BQML handles scaling automatically, implements Z-score standardization (the exact same thing as StandardScaler in scikit-learn)
) AS
SELECT
  -- We use the log-transformed features for better cluster stability
  log_monetary,
  log_frequency,
  log_recency
FROM
  `retail_segmentation.rfm_gold_2009_2010`;