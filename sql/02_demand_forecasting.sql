-- =============================================================================
-- 02_demand_forecasting.sql
-- Purpose: Real time-series demand forecasting using BigQuery ML (ARIMA_PLUS)
--          and ML.FORECAST across product categories
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

-- 2. Train Native BigQuery ML ARIMA_PLUS Time-Series Forecasting Model
CREATE OR REPLACE MODEL `aria_retail_intelligence.demand_forecast_arima`
OPTIONS(
    model_type = 'ARIMA_PLUS',
    time_series_timestamp_col = 'transaction_date',
    time_series_data_col = 'daily_units_sold',
    time_series_id_col = 'category',
    auto_arima = TRUE,
    data_frequency = 'DAILY',
    decompose_time_series = TRUE
) AS
SELECT
    transaction_date,
    category,
    daily_units_sold
FROM `aria_retail_intelligence.daily_category_demand`;

-- 3. Evaluate ARIMA_PLUS Forecast Accuracy Metrics
SELECT
    category,
    ROUND(aic, 2) AS aic,
    ROUND(variance, 4) AS variance,
    seasonal_periods
FROM ML.ARIMA_EVALUATE(MODEL `aria_retail_intelligence.demand_forecast_arima`)
ORDER BY aic ASC;

-- 4. Forecast Next 30 Days of Demand using ML.FORECAST
CREATE OR REPLACE TABLE `aria_retail_intelligence.category_demand_forecasts` AS
SELECT
    category,
    DATE(forecast_timestamp) AS forecast_date,
    ROUND(forecast_value, 1) AS predicted_units,
    ROUND(GREATEST(0, prediction_interval_lower_bound), 1) AS lower_bound_units,
    ROUND(prediction_interval_upper_bound, 1) AS upper_bound_units,
    ROUND(standard_error, 2) AS standard_error
FROM ML.FORECAST(
    MODEL `aria_retail_intelligence.demand_forecast_arima`,
    STRUCT(30 AS horizon, 0.90 AS confidence_level)
)
ORDER BY category, forecast_date;
