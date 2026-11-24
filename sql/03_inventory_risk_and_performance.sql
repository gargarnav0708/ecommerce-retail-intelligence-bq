-- =============================================================================
-- 03_inventory_risk_and_performance.sql
-- Purpose: Inventory stockout risk scoring (combining actual inventory_items
--          data with sales velocity) and BQML product performance prediction
-- =============================================================================

-- 1. Inventory Risk Analytics Pipeline (Joining inventory_items, transactions, products)
CREATE OR REPLACE TABLE `aria_retail_intelligence.inventory_risk_report` AS
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
    FROM `aria_retail_intelligence.transactions`
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
    -- Days of Supply = Available Stock / Daily Sales Velocity
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
FROM `aria_retail_intelligence.products` p
LEFT JOIN inventory_status i ON p.product_id = i.product_id
LEFT JOIN recent_velocity v ON p.product_id = v.product_id;

-- 2. Train Product Performance Prediction Model (BigQuery ML Regression)
CREATE OR REPLACE MODEL `aria_retail_intelligence.product_performance_model`
OPTIONS(
    model_type = 'LINEAR_REG',
    input_label_cols = ['revenue_30d'],
    ls_init_learn_rate = 0.15,
    max_iterations = 20
) AS
SELECT
    category,
    brand,
    department,
    retail_price,
    unit_cost,
    gross_margin,
    COALESCE(r.available_stock, 0) AS available_stock,
    COALESCE(r.units_sold_30d, 0) AS units_sold_30d,
    COALESCE(r.revenue_30d, 0) AS revenue_30d
FROM `aria_retail_intelligence.products` p
INNER JOIN `aria_retail_intelligence.inventory_risk_report` r
    USING (product_id)
WHERE r.revenue_30d > 0;

-- 3. Evaluate Product Performance Model
SELECT
    ROUND(mean_absolute_error, 2) AS mae,
    ROUND(mean_squared_error, 2) AS mse,
    ROUND(r2_score, 4) AS r2_score
FROM ML.EVALUATE(MODEL `aria_retail_intelligence.product_performance_model`);

-- 4. Predict Product Revenue Performance using ML.PREDICT
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
    MODEL `aria_retail_intelligence.product_performance_model`,
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
        FROM `aria_retail_intelligence.products` p
        INNER JOIN `aria_retail_intelligence.inventory_risk_report` r
            USING (product_id)
    )
)
ORDER BY predicted_revenue_30d DESC
LIMIT 25;
