"""
ARIA: E-Commerce Retail Intelligence Platform (BigQuery AI)
Clean Python wrapper for running the BigQuery SQL & AI pipelines.
"""

import os
import pandas as pd
from google.cloud import bigquery


class RetailIntelligencePipeline:
    """End-to-end BigQuery SQL & BigQuery AI pipeline using `thelook_ecommerce`."""

    def __init__(
        self,
        project_id: str | None = None,
        dataset_id: str = "aria_retail_intelligence",
        connection_id: str = "us.vertex_ai_conn",
        location: str = "US",
    ):
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT", "your-gcp-project-id")
        self.dataset_id = dataset_id
        self.connection_id = connection_id
        self.location = location
        self.dataset_ref = f"{self.project_id}.{self.dataset_id}"
        self.client = bigquery.Client(project=self.project_id, location=self.location)

    def ensure_dataset(self) -> None:
        """Create the BigQuery dataset if it does not exist."""
        dataset = bigquery.Dataset(self.dataset_ref)
        dataset.location = self.location
        self.client.create_dataset(dataset, exists_ok=True)
        print(f"[OK] Dataset ready: {self.dataset_ref}")

    def build_core_schema(self) -> pd.DataFrame:
        """
        Module 1: Build customers, products, and transactions tables from
        `bigquery-public-data.thelook_ecommerce` and run validation checks.
        """
        self.ensure_dataset()

        customers_sql = f"""
        CREATE OR REPLACE TABLE `{self.dataset_ref}.customers` AS
        SELECT
            u.id AS customer_id,
            u.age,
            u.gender,
            u.country,
            u.state,
            u.traffic_source,
            DATE(u.created_at) AS signup_date,
            COUNT(DISTINCT oi.order_id) AS total_orders,
            ROUND(COALESCE(SUM(oi.sale_price), 0), 2) AS lifetime_spend
        FROM `bigquery-public-data.thelook_ecommerce.users` u
        LEFT JOIN `bigquery-public-data.thelook_ecommerce.order_items` oi
            ON u.id = oi.user_id
            AND oi.status NOT IN ('Cancelled', 'Returned')
        GROUP BY u.id, u.age, u.gender, u.country, u.state, u.traffic_source, DATE(u.created_at)
        HAVING total_orders > 0
        LIMIT 5000
        """
        self.client.query(customers_sql).result()

        products_sql = f"""
        CREATE OR REPLACE TABLE `{self.dataset_ref}.products` AS
        SELECT
            p.id AS product_id,
            p.name AS product_name,
            p.category,
            p.brand,
            p.department,
            ROUND(p.cost, 2) AS unit_cost,
            ROUND(p.retail_price, 2) AS retail_price,
            ROUND(p.retail_price - p.cost, 2) AS gross_margin,
            CONCAT(
                'Product: ', COALESCE(p.name, 'Unknown'),
                ' | Brand: ', COALESCE(p.brand, 'Generic'),
                ' | Category: ', COALESCE(p.category, 'General'),
                ' | Department: ', COALESCE(p.department, 'General'),
                ' | Price: $', CAST(ROUND(p.retail_price, 2) AS STRING)
            ) AS content
        FROM `bigquery-public-data.thelook_ecommerce.products` p
        WHERE p.name IS NOT NULL
          AND p.retail_price > 0
        LIMIT 1000
        """
        self.client.query(products_sql).result()

        transactions_sql = f"""
        CREATE OR REPLACE TABLE `{self.dataset_ref}.transactions` AS
        SELECT
            oi.id AS transaction_id,
            oi.order_id,
            oi.user_id AS customer_id,
            oi.product_id,
            oi.inventory_item_id,
            oi.status AS order_status,
            DATE(oi.created_at) AS transaction_date,
            ROUND(oi.sale_price, 2) AS sale_price,
            ROUND(p.cost, 2) AS product_cost,
            ROUND(oi.sale_price - p.cost, 2) AS net_profit,
            p.category,
            p.brand
        FROM `bigquery-public-data.thelook_ecommerce.order_items` oi
        INNER JOIN `bigquery-public-data.thelook_ecommerce.products` p
            ON oi.product_id = p.id
        WHERE oi.status NOT IN ('Cancelled', 'Returned')
          AND oi.created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 365 DAY)
        """
        self.client.query(transactions_sql).result()

        validation_sql = f"""
        SELECT
            'customers' AS table_name,
            COUNT(*) AS total_rows,
            COUNT(DISTINCT customer_id) AS unique_primary_keys,
            COUNTIF(customer_id IS NULL) AS null_key_count,
            ROUND(AVG(lifetime_spend), 2) AS avg_metric_value
        FROM `{self.dataset_ref}.customers`
        UNION ALL
        SELECT
            'products' AS table_name,
            COUNT(*) AS total_rows,
            COUNT(DISTINCT product_id) AS unique_primary_keys,
            COUNTIF(product_id IS NULL OR content IS NULL) AS null_key_count,
            ROUND(AVG(retail_price), 2) AS avg_metric_value
        FROM `{self.dataset_ref}.products`
        UNION ALL
        SELECT
            'transactions' AS table_name,
            COUNT(*) AS total_rows,
            COUNT(DISTINCT transaction_id) AS unique_primary_keys,
            COUNTIF(transaction_id IS NULL OR sale_price <= 0) AS null_key_count,
            ROUND(AVG(sale_price), 2) AS avg_metric_value
        FROM `{self.dataset_ref}.transactions`
        """
        return self.client.query(validation_sql).to_dataframe()

    def run_demand_forecasting(self, horizon: int = 30) -> pd.DataFrame:
        """
        Module 2: Forecast 30-day category demand using BigQuery's native
        AI.FORECAST function powered by Google's TimesFM foundation model.
        """
        daily_demand_sql = f"""
        CREATE OR REPLACE TABLE `{self.dataset_ref}.daily_category_demand` AS
        SELECT
            transaction_date,
            category,
            COUNT(transaction_id) AS daily_units_sold,
            ROUND(SUM(sale_price), 2) AS daily_revenue
        FROM `{self.dataset_ref}.transactions`
        WHERE category IN (
            SELECT category
            FROM `{self.dataset_ref}.transactions`
            GROUP BY category
            ORDER BY COUNT(*) DESC
            LIMIT 6
        )
        GROUP BY transaction_date, category
        """
        self.client.query(daily_demand_sql).result()

        forecast_sql = f"""
        SELECT
            category,
            DATE(forecast_timestamp) AS forecast_date,
            ROUND(forecast_value, 1) AS predicted_units,
            ROUND(GREATEST(0, prediction_interval_lower_bound), 1) AS lower_bound_units,
            ROUND(prediction_interval_upper_bound, 1) AS upper_bound_units,
            confidence_level
        FROM AI.FORECAST(
            TABLE `{self.dataset_ref}.daily_category_demand`,
            data_col => 'daily_units_sold',
            timestamp_col => 'transaction_date',
            id_cols => ['category'],
            horizon => {horizon},
            confidence_level => 0.90
        )
        ORDER BY category, forecast_date
        """
        return self.client.query(forecast_sql).to_dataframe()

    def run_inventory_risk_and_performance(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Module 3: Compute SKU-level Days of Supply & Inventory Risk Tiers,
        and train a BQML Linear Regression model for product performance prediction.
        """
        inventory_risk_sql = f"""
        CREATE OR REPLACE TABLE `{self.dataset_ref}.inventory_risk_report` AS
        WITH inventory_status AS (
            SELECT
                product_id,
                COUNTIF(sold_at IS NULL) AS available_stock,
                COUNTIF(sold_at IS NOT NULL) AS historical_sold_units
            FROM `bigquery-public-data.thelook_ecommerce.inventory_items`
            GROUP BY product_id
        ),
        recent_velocity AS (
            SELECT
                product_id,
                COUNT(transaction_id) AS units_sold_30d,
                ROUND(COUNT(transaction_id) / 30.0, 2) AS daily_sales_velocity,
                ROUND(SUM(sale_price), 2) AS revenue_30d
            FROM `{self.dataset_ref}.transactions`
            WHERE transaction_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
            GROUP BY product_id
        )
        SELECT
            p.product_id,
            p.product_name,
            p.category,
            p.brand,
            p.retail_price,
            COALESCE(i.available_stock, 0) AS available_stock,
            COALESCE(v.units_sold_30d, 0) AS units_sold_30d,
            COALESCE(v.daily_sales_velocity, 0) AS daily_sales_velocity,
            COALESCE(v.revenue_30d, 0) AS revenue_30d,
            ROUND(
                SAFE_DIVIDE(COALESCE(i.available_stock, 0), NULLIF(v.daily_sales_velocity, 0)),
                1
            ) AS days_of_supply,
            CASE
                WHEN COALESCE(i.available_stock, 0) = 0 AND COALESCE(v.units_sold_30d, 0) > 0
                    THEN 'CRITICAL: Stockout'
                WHEN SAFE_DIVIDE(i.available_stock, NULLIF(v.daily_sales_velocity, 0)) < 14
                    THEN 'HIGH RISK: < 14 Days Supply'
                WHEN SAFE_DIVIDE(i.available_stock, NULLIF(v.daily_sales_velocity, 0)) < 30
                    THEN 'MODERATE RISK: 14-30 Days Supply'
                WHEN COALESCE(v.units_sold_30d, 0) = 0 AND COALESCE(i.available_stock, 0) > 20
                    THEN 'OVERSTOCK: Slow Mover'
                ELSE 'HEALTHY'
            END AS inventory_risk_tier
        FROM `{self.dataset_ref}.products` p
        LEFT JOIN inventory_status i ON p.product_id = i.product_id
        LEFT JOIN recent_velocity v ON p.product_id = v.product_id
        """
        self.client.query(inventory_risk_sql).result()

        perf_model_sql = f"""
        CREATE OR REPLACE MODEL `{self.dataset_ref}.product_performance_model`
        OPTIONS(
            model_type = 'LINEAR_REG',
            input_label_cols = ['revenue_30d'],
            ls_init_learn_rate = 0.15,
            max_iterations = 20
        ) AS
        SELECT
            p.category,
            p.brand,
            p.department,
            p.retail_price,
            p.unit_cost,
            p.gross_margin,
            COALESCE(r.available_stock, 0) AS available_stock,
            COALESCE(r.units_sold_30d, 0) AS units_sold_30d,
            COALESCE(r.revenue_30d, 0) AS revenue_30d
        FROM `{self.dataset_ref}.products` p
        INNER JOIN `{self.dataset_ref}.inventory_risk_report` r
            USING (product_id)
        WHERE r.revenue_30d > 0
        """
        self.client.query(perf_model_sql).result()

        eval_df = self.client.query(
            f"SELECT * FROM ML.EVALUATE(MODEL `{self.dataset_ref}.product_performance_model`)"
        ).to_dataframe()

        pred_sql = f"""
        SELECT
            product_id,
            product_name,
            category,
            brand,
            retail_price,
            units_sold_30d,
            revenue_30d AS actual_revenue_30d,
            ROUND(predicted_revenue_30d, 2) AS predicted_revenue_30d,
            inventory_risk_tier
        FROM ML.PREDICT(
            MODEL `{self.dataset_ref}.product_performance_model`,
            (
                SELECT
                    p.product_id,
                    p.product_name,
                    p.category,
                    p.brand,
                    p.department,
                    p.retail_price,
                    p.unit_cost,
                    p.gross_margin,
                    r.available_stock,
                    r.units_sold_30d,
                    r.revenue_30d,
                    r.inventory_risk_tier
                FROM `{self.dataset_ref}.products` p
                INNER JOIN `{self.dataset_ref}.inventory_risk_report` r
                    USING (product_id)
            )
        )
        ORDER BY predicted_revenue_30d DESC
        LIMIT 25
        """
        pred_df = self.client.query(pred_sql).to_dataframe()
        return eval_df, pred_df

    def register_remote_vertex_models(self) -> None:
        """
        Module 4a: Register Vertex AI Text Embedding (`text-embedding-004`) and
        Gemini (`gemini-1.5-flash`) remote models in BigQuery.
        """
        emb_model_sql = f"""
        CREATE OR REPLACE MODEL `{self.dataset_ref}.embedding_model`
        REMOTE WITH CONNECTION `{self.project_id}.{self.connection_id}`
        OPTIONS (ENDPOINT = 'text-embedding-004')
        """
        self.client.query(emb_model_sql).result()

        llm_model_sql = f"""
        CREATE OR REPLACE MODEL `{self.dataset_ref}.gemini_model`
        REMOTE WITH CONNECTION `{self.project_id}.{self.connection_id}`
        OPTIONS (ENDPOINT = 'gemini-1.5-flash')
        """
        self.client.query(llm_model_sql).result()

    def build_product_embeddings_and_search(self, query_text: str, top_k: int = 5) -> pd.DataFrame:
        """
        Module 4b: Generate real product embeddings using ML.GENERATE_EMBEDDING
        and retrieve top matches using native BigQuery VECTOR_SEARCH.
        """
        create_emb_sql = f"""
        CREATE OR REPLACE TABLE `{self.dataset_ref}.product_embeddings` AS
        SELECT
            product_id,
            product_name,
            category,
            brand,
            retail_price,
            content,
            ml_generate_embedding_result AS embedding
        FROM ML.GENERATE_EMBEDDING(
            MODEL `{self.dataset_ref}.embedding_model`,
            TABLE `{self.dataset_ref}.products`,
            STRUCT(TRUE AS flatten_json_output, 'RETRIEVAL_DOCUMENT' AS task_type)
        )
        WHERE ARRAY_LENGTH(ml_generate_embedding_result) > 0
        """
        self.client.query(create_emb_sql).result()

        vector_search_sql = f"""
        SELECT
            query.query_text,
            base.product_id,
            base.product_name,
            base.category,
            base.brand,
            base.retail_price,
            ROUND(1 - distance, 4) AS cosine_similarity
        FROM VECTOR_SEARCH(
            TABLE `{self.dataset_ref}.product_embeddings`,
            'embedding',
            (
                SELECT
                    @query_text AS query_text,
                    ml_generate_embedding_result AS embedding
                FROM ML.GENERATE_EMBEDDING(
                    MODEL `{self.dataset_ref}.embedding_model`,
                    (SELECT @query_text AS content),
                    STRUCT(TRUE AS flatten_json_output, 'RETRIEVAL_QUERY' AS task_type)
                )
            ),
            top_k => {top_k},
            distance_type => 'COSINE'
        )
        ORDER BY cosine_similarity DESC
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("query_text", "STRING", query_text)]
        )
        return self.client.query(vector_search_sql, job_config=job_config).to_dataframe()

    def generate_ai_restock_briefs(self, limit: int = 5) -> pd.DataFrame:
        """
        Module 4c: Generate executive merchandising & restock action plans for
        high-risk SKUs using ML.GENERATE_TEXT.
        """
        gen_text_sql = f"""
        SELECT
            product_id,
            product_name,
            category,
            inventory_risk_tier,
            ml_generate_text_llm_result AS ai_merchandising_action
        FROM ML.GENERATE_TEXT(
            MODEL `{self.dataset_ref}.gemini_model`,
            (
                SELECT
                    product_id,
                    product_name,
                    category,
                    inventory_risk_tier,
                    CONCAT(
                        'You are an e-commerce retail analyst. Write a concise 2-sentence action plan for ',
                        product_name, ' (Category: ', category, ', Price: $', CAST(retail_price AS STRING), '). ',
                        'Current stock: ', CAST(available_stock AS STRING), ' units, ',
                        '30-day sales: ', CAST(units_sold_30d AS STRING), ' units, ',
                        'Risk Status: ', inventory_risk_tier, '.'
                    ) AS prompt
                FROM `{self.dataset_ref}.inventory_risk_report`
                WHERE inventory_risk_tier IN ('CRITICAL: Stockout', 'HIGH RISK: < 14 Days Supply')
                ORDER BY revenue_30d DESC
                LIMIT {limit}
            ),
            STRUCT(
                0.2 AS temperature,
                150 AS max_output_tokens,
                TRUE AS flatten_json_output
            )
        )
        """
        return self.client.query(gen_text_sql).to_dataframe()
