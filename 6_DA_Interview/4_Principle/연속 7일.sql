-- 12월 기간 중 연속 7일을 머무를 수 있는 숙소를 찾고, 그 중 가장 저렴한 7일을 찾는 쿼리

-- 1단계: 각 숙소별로 날짜에 순번을 부여
WITH ranked AS (
  SELECT 
    house_id,
    date,
    price,
    ROW_NUMBER() OVER (PARTITION BY house_id ORDER BY date) AS rn
  FROM availability
  WHERE date BETWEEN '2023-12-01' AND '2023-12-31' -- 12월 기간으로 필터링
),

-- 2단계: 연속된 날짜를 그룹화하기 위한 키 생성
-- 날짜에서 순번을 빼면 연속된 날짜는 동일한 값(grp_key)을 가짐
grouped AS (
  SELECT 
    house_id,
    date,
    price,
    DATE_SUB(date, INTERVAL rn DAY) AS grp_key
  FROM ranked
),

-- 3단계: 연속된 날짜 그룹별로 시작일, 종료일, 일수, 총 가격 계산
-- 7일 이상 연속으로 이용 가능한 숙소만 필터링
grouped_with_sum AS (
  SELECT 
    house_id,
    grp_key,
    MIN(date) AS start_date,  -- 연속 기간의 시작일
    MAX(date) AS end_date,    -- 연속 기간의 종료일
    COUNT(*) AS days,         -- 연속된 일수
    SUM(price) AS total_price -- 해당 기간의 총 가격
  FROM grouped
  GROUP BY house_id, grp_key
  HAVING days >= 7  -- 7일 이상 연속으로 이용 가능한 경우만 선택
),

-- 4단계: 정확히 7일 기간에 대한 옵션 생성
-- 연속된 기간이 7일보다 길 경우, 처음 7일만 고려
seven_day_options AS (
  SELECT 
    house_id,
    start_date,
    DATE_ADD(start_date, INTERVAL 6 DAY) AS end_date,  -- 시작일로부터 6일 후 = 7일 기간
    SUM(price) AS total_price  -- 7일 기간의 총 가격
  FROM availability a
  JOIN grouped_with_sum g
    ON a.house_id = g.house_id
   AND a.date BETWEEN g.start_date AND DATE_ADD(g.start_date, INTERVAL 6 DAY)  -- 정확히 7일 기간만 조인
  GROUP BY house_id, start_date
)

-- 최종 결과: 가장 저렴한 7일 연속 숙박 옵션 선택
SELECT *
FROM seven_day_options
ORDER BY total_price  -- 가격이 가장 저렴한 순으로 정렬
LIMIT 1;  -- 가장 저렴한 옵션 1개만 선택
