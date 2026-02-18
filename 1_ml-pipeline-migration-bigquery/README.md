# Cloud-Native Customer Segmentation: BigQuery ML Pipeline

This project demonstrates the migration of a local Python-based K-means clustering pipeline to a production-scalable architecture using **Google Cloud BigQuery** and **BigQuery ML**.

## Architecture: The Medallion Pattern

We implemented a **Medallion Architecture** to ensure data quality and lineage throughout the ML lifecycle.

```mermaid
graph LR
    A[(Excel Local)] -->|Ingestion| B[Bronze: Raw Data]
    B -->|ETL SQL| C[Silver: Cleaned & Filtered]
    C -->|Feature Eng| D[Gold: RFM Metrics]
    D -->|BQML| E[K-Means Model]
    E -->|Predict| F[Final Business Segments]
```

### 1. Ingestion Layer (Bronze)
- **Tool**: `src/bq_ingestion.py`
- **Action**: Automated upload of multi-sheet retail data from Excel to BigQuery staging tables.

### 2. Silver Layer (Cleaned)
- **Tool**: `src/sql/etl.sql`
- **Logics**: Standardized schema, removal of cancellations, and regex-based validation of `Invoice` and `StockCode` patterns, mirroring the original Python ETL logic.

### 3. Gold Layer (Curated Features)
- **Tool**: `src/sql/rfm.sql`
- **Features**: Aggregated Recency, Frequency, and Monetary (RFM) metrics. 
- **Transformations**: Applied Log-transformations directly in SQL to manage data skewness for optimal K-means performance.

### 4. Machine Learning Layer (BQML)
- **Tool**: `src/sql/model_training.sql`
- **Model**: K-Means clustering trained directly on the 2009-2010 dataset.
- **Automation**: Automatic feature standardization (Z-score) handled by the data warehouse.

## Orchestration & Deployment
The entire pipeline is orchestrated via **`src/bq_pipeline.py`**, which manages dependencies and executes the SQL transformation sequence.

```bash
# Run the end-to-end pipeline
python src/bq_pipeline.py
```

## Business Insights: Segment Drift Analysis
By training on 2009 data and scoring on 2011 data, the pipeline identifies **Segment Drift**. This allows the business to track how many customers are migrating from **'Champions'** to **'At Risk'** or **'Hibernating'** over time.

| Segment | 2009 Count | 2011 Count | % Change |
|---------|------------|------------|----------|
| Champions | ... | ... | ... |
| Loyal | ... | ... | ... |
| New / Promising | ... | ... | ... |
| Hibernating | ... | ... | ... |

---

## Tech Stack
- **Cloud**: Google Cloud Platform (GCP)
- **Data Warehouse**: BigQuery
- **Machine Learning**: BigQuery ML (K-Means)
- **Orchestration**: Python (Google Cloud SDK)
- **Environment**: Python-dotenv, Pandas-GBQ
