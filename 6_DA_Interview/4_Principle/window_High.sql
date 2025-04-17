
-- Window Function 범위 지정 옵션 예시

-- 1. ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
-- 파티션의 첫 행부터 현재 행까지의 모든 행을 포함
SELECT 
    order_date,
    customer_id,
    order_amount,
    SUM(order_amount) OVER (
        PARTITION BY customer_id 
        ORDER BY order_date 
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS running_total
FROM orders;

-- 2. ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
-- 파티션 내의 모든 행을 포함 (전체 파티션)
SELECT 
    product_id,
    category,
    price,
    AVG(price) OVER (
        PARTITION BY category 
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS category_avg_price
FROM products;

-- 3. ROWS BETWEEN n PRECEDING AND CURRENT ROW
-- 현재 행 기준 이전 n개 행과 현재 행을 포함
SELECT 
    date,
    stock_price,
    AVG(stock_price) OVER (
        ORDER BY date 
        ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
    ) AS moving_avg_4days
FROM stock_prices;

-- 4. ROWS BETWEEN CURRENT ROW AND n FOLLOWING
-- 현재 행부터 이후 n개 행을 포함
SELECT 
    visit_date,
    page_views,
    SUM(page_views) OVER (
        ORDER BY visit_date 
        ROWS BETWEEN CURRENT ROW AND 2 FOLLOWING
    ) AS next_3days_views
FROM website_traffic;

-- 5. ROWS BETWEEN n PRECEDING AND n FOLLOWING
-- 현재 행 기준 이전 n개, 이후 n개, 그리고 현재 행을 포함 (슬라이딩 윈도우)
SELECT 
    transaction_date,
    amount,
    AVG(amount) OVER (
        ORDER BY transaction_date 
        ROWS BETWEEN 2 PRECEDING AND 2 FOLLOWING
    ) AS centered_5day_avg
FROM transactions;

-- 6. RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
-- ORDER BY 값이 같은 행들을 그룹화하여 처리 (ROWS와 다름)
SELECT 
    order_date,
    region,
    sales,
    SUM(sales) OVER (
        PARTITION BY region 
        ORDER BY order_date 
        RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cumulative_sales_by_date
FROM sales_data;

-- 7. 기본값 (RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
-- ORDER BY만 지정하고 범위를 지정하지 않으면 위와 같은 기본값 적용
SELECT 
    employee_id,
    department,
    salary,
    RANK() OVER (
        PARTITION BY department 
        ORDER BY salary DESC
    ) AS salary_rank
FROM employees;
