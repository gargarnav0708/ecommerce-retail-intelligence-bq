-- =============================================================================
-- 02_demand_forecasting.sql
-- Purpose: Zero-shot deep-learning demand forecasting using BigQuery's native
--          AI.FORECAST function powered by Google Research's TimesFM
--          (Time-Series Foundation Model)
-- =============================================================================

-- 1. Build Daily Category Demand Time-Series Table
CREATE OR REPLACE TABLE `aria_retail_intelligence.daily_category_demand` AS
SELECT
    transaction_date,
    category,
    COUNT(transaction_id) AS daily_units_sold,
    ROUND(SUM(sale_price), 2) AS daily_revenue
FROM `aria_retail_intelligence.transactions`
WHERE category IN (
    SELECT category
    FROM `aria_retail_intelligence.transactions`
    GROUP BY category
    ORDER BY COUNT(*) DESC
    LIMIT 6
)
GROUP BY transaction_date, category;

-- 2. Forecast Next 30 Days of Demand using BigQuery AI.FORECAST (TimesFM)
-- No manual model training (CREATE MODEL) required: TimesFM is a pretrained
-- decoder-only Transformer foundation model built directly into BigQuery AI.
CREATE OR REPLACE TABLE `aria_retail_intelligence.category_demand_forecasts` AS
SELECT
    category,
    DATE(forecast_timestamp) AS forecast_date,
    ROUND(forecast_value, 1) AS predicted_units,
    ROUND(GREATEST(0, prediction_interval_lower_bound), 1) AS lower_bound_units,
    ROUND(prediction_interval_upper_bound, 1) AS upper_bound_units,
    confidence_level
FROM AI.FORECAST(
    TABLE `aria_retail_intelligence.daily_category_demand`,
    data_col => 'daily_units_sold',
    timestamp_col => 'transaction_date',
    id_cols => ['category'],
    horizon => 30,
    confidence_level => 0.90
)
ORDER BY category, forecast_date;

-- 3. Aggregate 30-Day Forecasted Demand Summary by Category
SELECT
    category,
    COUNT(forecast_date) AS forecast_horizon_days,
    ROUND(SUM(predicted_units), 0) AS total_predicted_units_30d,
    ROUND(AVG(predicted_units), 1) AS avg_daily_predicted_units,
    ROUND(SUM(lower_bound_units), 0) AS conservative_90pct_floor,
    ROUND(SUM(upper_bound_units), 0) AS peak_90pct_ceiling
FROM `aria_retail_intelligence.category_demand_forecasts`
GROUP BY category
ORDER BY total_predicted_units_30d DESC;
