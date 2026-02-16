# Research & Planning Plan: BigQuery K-means Pipeline

This document outlines the necessary research and preparation steps to transition the current K-means clustering pipeline to Google Cloud BigQuery and BigQuery ML (BQML).

## Proposed File Structure & Resources

To maintain a professional portfolio, we will structure the code as follows:

### 1. Python Orchestration (`src/`)
- **`bq_ingestion.py`**: Script to upload local data (Excel/PKL) to BigQuery `staging` tables.
- **`bq_pipeline.py`**: Main orchestrator. Triggers SQL transformations and BQML model training.
- **`utils_bq.py`**: Helper functions for BigQuery clients, error handling, and authentication.

### 2. SQL Feature Engineering (`src/sql/`)
- **`rfm_cleaning.sql`**: Data cleaning scripts (removing cancellations, invalid codes).
- **`rfm_aggregation.sql`**: SQL logic to calculate Recency, Frequency, and Monetary metrics.
- **`model_training.sql`**: The `CREATE MODEL` statement for BQML K-means.
- **`scoring_comparison.sql`**: SQL for predicting clusters and generating comparison metrics.

### 3. Analysis & Visuals (`notebooks/`)
- **`bq_analysis.ipynb`**: Fetches result tables from BigQuery and generates final plots for the portfolio.

---

## Repository Strategy

> [!TIP]
> **Recommendation: Create a NEW Repository**
> Instead of keeping this inside the `ds_projects_collabs` folder, I recommend creating a dedicated repository named something like `ML-Pipeline-Migration-BigQuery`.
> 
> **Why?**
> 1. **Focus**: It allows the `README.md` to focus exclusively on the "Local-to-Cloud" narrative.
> 2. **Visibility**: Recruiters can immediately see a "Cloud Engineer / Data Scientist" hybrid project.
> 3. **Cleanliness**: Avoids cluttering a general-purpose project with specific cloud configurations.

---

## Research Objectives

Before implementation, we need to clarify and research the following areas:

### 1. Google Cloud Environment & Auth
- **GCP Project Setup**: Identify or create a Google Cloud Project and Dataset.
- **Local Auth**: Research how to authenticate your local environment (Service Account vs. ADC).
- **Libraries**: Verify compatibility of `google-cloud-bigquery` and `pandas-gbq`.

### 2. BigQuery ML (BQML) K-Means
- **Standardization**: Research if BQML's K-means automatically scales features.
- **Hyperparameters**: Research BQML options for choosing `k` (e.g., auto-tuning).

### 3. SQL-Based Feature Engineering
- **Cleaning Logic**: Map Python cleaning steps (regex, filters) to BigQuery SQL.
- **RFM Generation**: Draft queries for aggregation and `log1p` transformation.

### 4. Cluster Comparison Strategy
- **Segment Shifting**: Research methodologies for comparing clusters across two time periods.
- **Scoring**: Understand how `ML.PREDICT` works for "future" data scoring.

## Proposed Strategy (Research Phase)

1. **GCP Sandbox**: Create a trial dataset in BigQuery.
2. **SQL Prototyping**: Test cleaning and RFM logic directly in the BigQuery Console.
3. **Connectivity Test**: Run a simple Python script to verify access.

## Next Steps

> [!NOTE]
> Once this research phase is complete, we will proceed to create a detailed **Execution Plan** for the actual implementation.
