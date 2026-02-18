-- 1. Score the Training Data (2009-2010)
CREATE OR REPLACE TABLE `retail_segmentation.scored_2009_2010` AS
SELECT
  *
FROM
  ML.PREDICT(MODEL `retail_segmentation.customer_segments_model`,
    (SELECT * FROM `retail_segmentation.rfm_gold_2009_2010`));

-- 2. Score the Testing/Future Data (2010-2011)
CREATE OR REPLACE TABLE `retail_segmentation.scored_2010_2011` AS
SELECT
  *
FROM
  ML.PREDICT(MODEL `retail_segmentation.customer_segments_model`,
    (SELECT * FROM `retail_segmentation.rfm_gold_2010_2011`));