-- =============================================================================
-- 04_bigquery_ai_semantic_and_genai.sql
-- Purpose: Real BigQuery AI functions — ML.GENERATE_EMBEDDING, VECTOR_SEARCH,
--          and ML.GENERATE_TEXT (via Vertex AI Remote Connection in BigQuery)
-- =============================================================================
-- Prerequisite: A BigQuery Cloud Resource connection (e.g. `us.vertex_ai_conn`)
-- granted the "Vertex AI User" IAM role.

-- 1. Register Remote Vertex AI Text Embedding Model in BigQuery
CREATE OR REPLACE MODEL `aria_retail_intelligence.embedding_model`
REMOTE WITH CONNECTION `us.vertex_ai_conn`
OPTIONS (ENDPOINT = 'text-embedding-004');

-- 2. Register Remote Vertex AI Gemini LLM in BigQuery
CREATE OR REPLACE MODEL `aria_retail_intelligence.gemini_model`
REMOTE WITH CONNECTION `us.vertex_ai_conn`
OPTIONS (ENDPOINT = 'gemini-1.5-flash');

-- 3. Generate Vector Embeddings for Products Catalog using ML.GENERATE_EMBEDDING
CREATE OR REPLACE TABLE `aria_retail_intelligence.product_embeddings` AS
SELECT
    product_id,
    product_name,
    category,
    brand,
    retail_price,
    content,
    ml_generate_embedding_result AS embedding
FROM ML.GENERATE_EMBEDDING(
    MODEL `aria_retail_intelligence.embedding_model`,
    TABLE `aria_retail_intelligence.products`,
    STRUCT(TRUE AS flatten_json_output, 'RETRIEVAL_DOCUMENT' AS task_type)
)
WHERE ARRAY_LENGTH(ml_generate_embedding_result) > 0;

-- 4. Semantic Product Discovery using Native BigQuery VECTOR_SEARCH
SELECT
    query.query_text,
    base.product_id,
    base.product_name,
    base.category,
    base.brand,
    base.retail_price,
    ROUND(1 - distance, 4) AS cosine_similarity
FROM VECTOR_SEARCH(
    TABLE `aria_retail_intelligence.product_embeddings`,
    'embedding',
    (
        SELECT
            'comfortable breathable athletic running jacket for outdoor workouts' AS query_text,
            ml_generate_embedding_result AS embedding
        FROM ML.GENERATE_EMBEDDING(
            MODEL `aria_retail_intelligence.embedding_model`,
            (SELECT 'comfortable breathable athletic running jacket for outdoor workouts' AS content),
            STRUCT(TRUE AS flatten_json_output, 'RETRIEVAL_QUERY' AS task_type)
        )
    ),
    top_k => 5,
    distance_type => 'COSINE'
)
ORDER BY cosine_similarity DESC;

-- 5. Automated Inventory Restock & Merchandising Briefs using ML.GENERATE_TEXT
SELECT
    product_id,
    product_name,
    category,
    inventory_risk_tier,
    ml_generate_text_llm_result AS ai_merchandising_action
FROM ML.GENERATE_TEXT(
    MODEL `aria_retail_intelligence.gemini_model`,
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
        FROM `aria_retail_intelligence.inventory_risk_report`
        WHERE inventory_risk_tier IN ('CRITICAL: Stockout', 'HIGH RISK: < 14 Days Supply')
        ORDER BY revenue_30d DESC
        LIMIT 10
    ),
    STRUCT(
        0.2 AS temperature,
        150 AS max_output_tokens,
        TRUE AS flatten_json_output
    )
);
