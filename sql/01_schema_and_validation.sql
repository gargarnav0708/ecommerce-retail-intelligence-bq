-- =============================================================================
-- 01_schema_and_validation.sql
-- Purpose: Extract, join, aggregate & validate customer, product & transaction
--          tables from `bigquery-public-data.thelook_ecommerce`
-- =============================================================================

-- 1. Create Curated Customers Table
CREATE OR REPLACE TABLE `aria_retail_intelligence.customers` AS
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
LIMIT 5000;

-- 2. Create Curated Products Table (with rich text content for ML.GENERATE_EMBEDDING)
CREATE OR REPLACE TABLE `aria_retail_intelligence.products` AS
SELECT
    p.id AS product_id,
    p.name AS product_name,
    p.category,
    p.brand,
    p.department,
    ROUND(p.cost, 2) AS unit_cost,
    ROUND(p.retail_price, 2) AS retail_price,
    ROUND(p.retail_price - p.cost, 2) AS gross_margin,
    -- `content` column is required by BigQuery ML.GENERATE_EMBEDDING
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
LIMIT 1000;

-- 3. Create Curated Transactions Table (Joining order_items, orders, products)
CREATE OR REPLACE TABLE `aria_retail_intelligence.transactions` AS
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
  AND oi.created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 365 DAY);

-- 4. Data Validation & Integrity Pipeline
SELECT
    'customers' AS table_name,
    COUNT(*) AS total_rows,
    COUNT(DISTINCT customer_id) AS unique_primary_keys,
    COUNTIF(customer_id IS NULL) AS null_key_count,
    ROUND(AVG(lifetime_spend), 2) AS avg_metric_value
FROM `aria_retail_intelligence.customers`

UNION ALL

SELECT
    'products' AS table_name,
    COUNT(*) AS total_rows,
    COUNT(DISTINCT product_id) AS unique_primary_keys,
    COUNTIF(product_id IS NULL OR content IS NULL) AS null_key_count,
    ROUND(AVG(retail_price), 2) AS avg_metric_value
FROM `aria_retail_intelligence.products`

UNION ALL

SELECT
    'transactions' AS table_name,
    COUNT(*) AS total_rows,
    COUNT(DISTINCT transaction_id) AS unique_primary_keys,
    COUNTIF(transaction_id IS NULL OR sale_price <= 0) AS null_key_count,
    ROUND(AVG(sale_price), 2) AS avg_metric_value
FROM `aria_retail_intelligence.transactions`;
