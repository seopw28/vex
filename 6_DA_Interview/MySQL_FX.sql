-- MySQL 코딩테스트 필수 함수 정리



-- 0. 실전

-- a : 퍼센타일을 매기고 내림차순(DESC)으로 정렬
 
percent_rank()over(order by size_of_colony desc) as p_rnk

-- b : dt to date

-- b : 문자열 형식의 날짜(YYYYMMDD)를 DATE 형식(YYYY-MM-DD)으로 변환

-- 방법 1: STR_TO_DATE 함수 사용
SELECT STR_TO_DATE('20230101', '%Y%m%d');  -- 결과: 2023-01-01

-- 방법 2: DATE_FORMAT과 STR_TO_DATE 조합
SELECT DATE_FORMAT(STR_TO_DATE('20230101', '%Y%m%d'), '%Y-%m-%d');  -- 결과: 2023-01-01
-- 날짜 형식 변환 (dt 컬럼이 datetime 타입일 경우)
select date_format(dt, '%Y-%m-%d') as dt_2 

-- 다른 방법들:
-- 1. 날짜만 추출
select cast(dt as date) as dt_2

-- 2. 다른 포맷으로 변환 (예: 월/일/년)
select date_format(dt, '%m/%d/%Y') as dt_2

-- 3. 연/월만 추출
select date_format(dt, '%Y-%m') as year_month
;

-- 방법 3: CONVERT 함수 사용
SELECT CONVERT(STR_TO_DATE('20230101', '%Y%m%d'), DATE);  -- 결과: 2023-01-01

;


-- 1. 날짜 함수
-- 날짜 추출 및 포맷팅
SELECT 
    NOW(),                                  -- 현재 날짜와 시간 (YYYY-MM-DD HH:MM:SS)
    CURDATE(),                              -- 현재 날짜 (YYYY-MM-DD)
    DATE_FORMAT(NOW(), '%Y-%m-%d'),         -- 날짜 포맷팅
    DATE_FORMAT(NOW(), '%Y-%m-%d %H:%i:%s') -- 날짜 시간 포맷팅
;

-- 날짜 계산
SELECT
    DATE_ADD(NOW(), INTERVAL 1 DAY),        -- 1일 후
    DATE_ADD(NOW(), INTERVAL 1 MONTH),      -- 1개월 후
    DATE_ADD(NOW(), INTERVAL 1 YEAR),       -- 1년 후
    DATE_SUB(NOW(), INTERVAL 7 DAY),        -- 7일 전
    NOW() + INTERVAL 1 DAY,                 -- 1일 후 (대체 문법)
    NOW() - INTERVAL 7 DAY                  -- 7일 전 (대체 문법)
;

-- 날짜 차이 계산
SELECT
    DATEDIFF('2023-12-31', '2023-01-01'),   -- 두 날짜 사이의 일수 차이
    TIMESTAMPDIFF(DAY, '2023-01-01', '2023-12-31'),    -- 일 단위 차이
    TIMESTAMPDIFF(MONTH, '2023-01-01', '2023-12-31'),  -- 월 단위 차이
    TIMESTAMPDIFF(YEAR, '2020-01-01', '2023-12-31')    -- 년 단위 차이
;

-- 날짜 부분 추출
SELECT
    YEAR(NOW()),       -- 연도 추출
    MONTH(NOW()),      -- 월 추출
    DAY(NOW()),        -- 일 추출
    HOUR(NOW()),       -- 시간 추출
    MINUTE(NOW()),     -- 분 추출
    SECOND(NOW()),     -- 초 추출
    DAYOFWEEK(NOW()),  -- 요일 추출 (1=일요일, 2=월요일, ..., 7=토요일)
    DAYOFMONTH(NOW()), -- 해당 월의 몇 번째 날인지
    DAYOFYEAR(NOW())   -- 해당 연도의 몇 번째 날인지
;

-- 날짜 반올림 (MySQL에는 DATE_TRUNC 함수가 없어 DATE_FORMAT으로 대체)
SELECT
    DATE_FORMAT(NOW(), '%Y-01-01'),                     -- 연도 시작일
    DATE_FORMAT(NOW(), '%Y-%m-01'),                     -- 월 시작일
    LAST_DAY(NOW()),                                    -- 월 마지막일
    DATE_FORMAT(DATE_SUB(NOW(), INTERVAL DAYOFWEEK(NOW())-1 DAY), '%Y-%m-%d') -- 주 시작일(월요일)
;

-- 2. 문자열 함수
SELECT
    CONCAT('Hello', ' ', 'World'),          -- 문자열 연결
    LENGTH('Hello World'),                  -- 문자열 길이
    UPPER('hello'),                         -- 대문자 변환
    LOWER('HELLO'),                         -- 소문자 변환
    SUBSTRING('Hello World', 1, 5),         -- 부분 문자열 추출
    REPLACE('Hello World', 'World', 'SQL'), -- 문자열 대체
    TRIM('  Hello  ')                       -- 공백 제거
;

-- 3. 집계 함수
SELECT
    COUNT(*),                               -- 행 수 계산
    SUM(column_name),                       -- 합계
    AVG(column_name),                       -- 평균
    MIN(column_name),                       -- 최소값
    MAX(column_name),                       -- 최대값
    GROUP_CONCAT(column_name)               -- 그룹 내 값들을 연결
FROM table_name;

-- 4. 윈도우 함수
SELECT
    ROW_NUMBER() OVER(PARTITION BY category ORDER BY sales DESC), -- 행 번호
    RANK() OVER(PARTITION BY category ORDER BY sales DESC),       -- 순위(동일 값은 동일 순위, 다음 순위는 건너뜀)
    DENSE_RANK() OVER(PARTITION BY category ORDER BY sales DESC), -- 순위(동일 값은 동일 순위, 다음 순위는 연속)
    LEAD(sales) OVER(PARTITION BY category ORDER BY date),        -- 다음 행 값
    LAG(sales) OVER(PARTITION BY category ORDER BY date)          -- 이전 행 값
FROM sales_table;

-- 5. 조건부 함수
SELECT
    IF(condition, true_value, false_value),                      -- 간단한 조건문
    CASE WHEN condition1 THEN result1
         WHEN condition2 THEN result2
         ELSE default_result
    END,                                                         -- 복잡한 조건문
    IFNULL(column_name, 'N/A'),                                  -- NULL 처리
    COALESCE(value1, value2, value3, ...)                        -- 첫 번째 NULL이 아닌 값 반환
FROM table_name;

-- 6. RFM 분석 관련 쿼리 예시 (Recency, Frequency, Monetary)
SELECT 
    customer_id,
    MAX(order_date) as last_purchase_date,                       -- Recency
    DATEDIFF(CURDATE(), MAX(order_date)) as days_since_last,
    COUNT(*) as purchase_count,                                  -- Frequency
    SUM(order_amount) as total_spent                             -- Monetary
FROM orders
GROUP BY customer_id;

-- 7. 데이터 분석용 쿼리 패턴
-- 월별 매출 추이
SELECT 
    DATE_FORMAT(order_date, '%Y-%m') as month,
    SUM(order_amount) as monthly_revenue,
    COUNT(DISTINCT customer_id) as unique_customers,
    SUM(order_amount)/COUNT(DISTINCT customer_id) as avg_revenue_per_customer
FROM orders
GROUP BY DATE_FORMAT(order_date, '%Y-%m')
ORDER BY month;

-- 8. 코호트 분석 기본 쿼리
SELECT 
    DATE_FORMAT(first_purchase_date, '%Y-%m') as cohort_month,
    TIMESTAMPDIFF(MONTH, first_purchase_date, order_date) as month_number,
    COUNT(DISTINCT customer_id) as customer_count
FROM (
    SELECT 
        customer_id,
        MIN(order_date) as first_purchase_date
    FROM orders
    GROUP BY customer_id
) first_purchases
JOIN orders ON orders.customer_id = first_purchases.customer_id
GROUP BY cohort_month, month_number
ORDER BY cohort_month, month_number;
