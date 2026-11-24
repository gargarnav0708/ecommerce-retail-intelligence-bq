# ARIA: E-Commerce Retail Intelligence Platform (BigQuery AI)

An end-to-end e-commerce data engineering, predictive analytics, and generative AI platform built on **Google Cloud BigQuery** using the free public dataset **`bigquery-public-data.thelook_ecommerce`**.

Unlike toy simulations that hardcode SQL string concatenations or fake vector arrays, this repository uses **real BigQuery ML and BigQuery AI (Vertex AI Remote Models)** functions directly in SQL:
- **`ARIMA_PLUS` & `ML.FORECAST`** for time-series demand forecasting with prediction intervals.
- **`LINEAR_REG`, `ML.EVALUATE` & `ML.PREDICT`** for SKU-level 30-day revenue performance prediction.
- **`ML.GENERATE_EMBEDDING` (`text-embedding-004`) & `VECTOR_SEARCH`** for 768-dimensional semantic product discovery.
- **`ML.GENERATE_TEXT` (`gemini-1.5-flash`)** for automated inventory restock and merchandising briefs.

---

## Repository Structure

```text
ecommerce-retail-intelligence-bq/
├── aria_bigquery_ai_ecommerce.ipynb          # Clean end-to-end Jupyter Notebook
├── sql/
│   ├── 01_schema_and_validation.sql          # Curated schema (customers, products, transactions) & validation
│   ├── 02_demand_forecasting.sql             # ARIMA_PLUS training, ML.ARIMA_EVALUATE & ML.FORECAST
│   ├── 03_inventory_risk_and_performance.sql # Days-of-Supply stockout risk & LINEAR_REG ML.PREDICT
│   └── 04_bigquery_ai_semantic_and_genai.sql # ML.GENERATE_EMBEDDING, VECTOR_SEARCH & ML.GENERATE_TEXT
├── src/
│   └── pipeline.py                           # Modular Python SDK runner for all 4 SQL/AI stages
├── requirements.txt                          # Python dependencies
└── README.md                                 # Project documentation
```

---

## Core Architecture & Modules

### 1. Schema Design, SQL Pipelines & Data Validation (`sql/01_schema_and_validation.sql`)
Transforms raw tables from `bigquery-public-data.thelook_ecommerce` (`users`, `products`, `order_items`, `inventory_items`) into three curated tables inside `aria_retail_intelligence`:
- **`customers`**: Joins user demographics with completed order line-items to compute `total_orders` and `lifetime_spend`.
- **`products`**: Extracts product metadata, cost, retail price, gross margin, and builds a standardized `content` column required by `ML.GENERATE_EMBEDDING`.
- **`transactions`**: Joins trailing 365-day completed order items with product cost and category metadata to track net profit per transaction.
- **Validation Pipeline**: Automated `UNION ALL` integrity check verifying row counts, primary key uniqueness (`COUNT(DISTINCT ...)`), null-key constraints (`COUNTIF`), and baseline metric averages.

### 2. Demand Forecasting (`sql/02_demand_forecasting.sql`)
- Aggregates daily unit sales by product category into `daily_category_demand`.
- Trains a multi-series BigQuery ML **`ARIMA_PLUS`** model (`demand_forecast_arima`) with `auto_arima = TRUE` and `decompose_time_series = TRUE`.
- Evaluates AIC and variance via **`ML.ARIMA_EVALUATE`** and forecasts 30-day demand with 90% confidence intervals using **`ML.FORECAST`**.

### 3. Inventory Risk & Product-Performance Prediction (`sql/03_inventory_risk_and_performance.sql`)
- **Inventory Risk Scoring**: Joins `inventory_items` (unsold stock where `sold_at IS NULL`) with 30-day sales velocity to compute **Days of Supply (DoS)** via `SAFE_DIVIDE`, classifying every SKU into `CRITICAL: Stockout`, `HIGH RISK: < 14 Days Supply`, `MODERATE RISK: 14-30 Days Supply`, `OVERSTOCK: Slow Mover`, or `HEALTHY`.
- **Product Performance Model**: Trains a BigQuery ML **`LINEAR_REG`** model (`product_performance_model`) predicting 30-day revenue from category, brand, price, gross margin, available stock, and unit velocity, evaluated with **`ML.EVALUATE`** (`MAE`, `MSE`, $R^2$) and scored with **`ML.PREDICT`**.

### 4. Semantic Discovery & Generative AI (`sql/04_bigquery_ai_semantic_and_genai.sql`)
- **Vector Embeddings (`ML.GENERATE_EMBEDDING`)**: Connects BigQuery to Vertex AI `text-embedding-004` to encode the `products` catalog into vector embeddings (`product_embeddings`).
- **Semantic Search (`VECTOR_SEARCH`)**: Encodes natural-language shopper queries on the fly (`task_type => 'RETRIEVAL_QUERY'`) and retrieves the top-$K$ closest products using `distance_type => 'COSINE'`.
- **Generative Merchandising (`ML.GENERATE_TEXT`)**: Calls Vertex AI `gemini-1.5-flash` directly from SQL over high-risk SKUs in `inventory_risk_report` to generate actionable 2-sentence restock and pricing plans.

---

## Quickstart & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Google Cloud & BigQuery Connection
Authenticate with Google Cloud and set your project ID:
```bash
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
```

*(For Module 4 only)* Create a BigQuery Cloud Resource Connection for Vertex AI remote models and grant its service account the **Vertex AI User** (`roles/aiplatform.user`) role:
```bash
bq mk --connection --location=US --project_id=$GOOGLE_CLOUD_PROJECT \
    --connection_type=CLOUD_RESOURCE vertex_ai_conn
```

### 3. Run the Notebook or Python Pipeline
Open [`aria_bigquery_ai_ecommerce.ipynb`](./aria_bigquery_ai_ecommerce.ipynb) in Jupyter/Colab/Kaggle, or execute via Python:
```python
from src.pipeline import RetailIntelligencePipeline

pipeline = RetailIntelligencePipeline(project_id="your-gcp-project-id")
validation_df = pipeline.build_core_schema()
forecast_df = pipeline.run_demand_forecasting(horizon=30)
eval_df, predictions_df = pipeline.run_inventory_risk_and_performance()
```
