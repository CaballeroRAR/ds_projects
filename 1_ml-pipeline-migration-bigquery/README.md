# ML Pipeline Migration: Local Python to Google Cloud BigQuery

This project demonstrates the migration of a local, memory-bound machine learning pipeline to a scalable, cloud-native architecture using **Google Cloud BigQuery** and **BigQuery ML (BQML)**.

## Project Evolution

### Phase 1: Local Implementation (Baseline)
The original project, located in the [1-cluster_retail_uci](https://github.com/CaballeroRAR/ds_projects_collabs/tree/main/1-cluster_retail_uci) repository, implements an end-to-end customer segmentation pipeline:
- **Environment**: Local Python (Pandas, Scikit-Learn).
- **Data Source**: UCI Online Retail II dataset.
- **Features**: RFM (Recency, Frequency, Monetary) analysis.
- **Model**: K-Means clustering.
- **Limitation**: Memory-bound, manual execution, and local-only state management.

### Phase 2: Cloud Migration (This Project)
We are evolving this pipeline into a production-ready cloud architecture. This transition showcases the ability to handle larger datasets and leverage managed cloud services for both data processing and machine learning.

## Tech Stack

- **Data Warehouse**: Google Cloud BigQuery
- **Machine Learning**: BigQuery ML (BQML) - K-Means
- **Language**: Python (Orchestration) & SQL (ETL/Feature Engineering)
- **SDKs**: `google-cloud-bigquery`, `pandas-gbq`
- **Authentication**: Google Cloud Application Default Credentials (ADC)

## Migration Strategy

1. **Scalable Ingestion**: Move from local `.pkl` or Excel files to BigQuery tables.
2. **SQL-Native ETL**: Replace Pandas cleaning logic with SQL views and scripts for better performance on large data.
3. **In-Warehouse Training**: Use BQML to train K-Means models directly where the data resides, eliminating the need to move data to a local compute environment.
4. **Drift Analysis**: Use BigQuery's analytical power to compare customer segments across different years (2009-2010 vs. 2010-2011) and detect behavioral shifts.

## Planned Structure

```text
├── src/
│   ├── bq_ingestion.py    # Data upload to GCP
│   ├── bq_pipeline.py     # Orchestration logic
│   └── sql/
│       ├── rfm_prep.sql   # SQL-based Feature Engineering
│       └── model_train.sql # BQML Training logic
├── notebooks/
│   └── bq_analysis.ipynb  # Final results and visualizations
└── README.md              # Project documentation
```

## Current Status: Research & Planning
We are currently in the research phase, focusing on:
- Mapping local Pandas cleaning logic to BigQuery SQL.
- Researching BQML's automated feature scaling capabilities.
- Defining the methodology for segment shift detection between datasets.
