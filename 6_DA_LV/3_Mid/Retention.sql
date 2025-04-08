WITH user_activity AS (
    -- 각 사용자별 첫 활동일과 모든 활동일을 추출
    -- first_date: 사용자의 최초 활동일
    -- activity_date: 사용자의 모든 활동일
    SELECT 
        user_id,
        DATE(MIN(event_time)) as first_date,
        DATE(event_time) as activity_date
    FROM user_events
    GROUP BY 1, 3
),

cohort_size AS (
    -- 코호트 크기 계산
    -- 각 최초 활동일(first_date)별 순수 사용자 수 집계
    SELECT 
        first_date,
        COUNT(DISTINCT user_id) as users
    FROM user_activity
    GROUP BY 1
),

retention_data AS (
    -- 리텐션 데이터 계산
    -- 최초 활동일과 이후 활동들 간의 일수 차이 계산
    -- active_users: 각 시점별 활성 사용자 수
    -- day_number: 최초 활동일로부터 경과된 일수
    SELECT
        a.first_date,
        a.activity_date,
        COUNT(DISTINCT a.user_id) as active_users,
        DATE_DIFF('day', a.first_date, a.activity_date) as day_number
    FROM user_activity a
    GROUP BY 1, 2 -- 4는 1,2 에 의한 고정값을 갖게 됨.
)

-- 최종 리텐션 데이터 출력
-- cohort_date: 코호트의 시작일
-- cohort_size: 해당 코호트의 전체 사용자 수
-- day_number: 경과 일수
-- active_users: 해당 일의 활성 사용자 수
-- retention_rate: 리텐션율(%) = (활성 사용자 수 / 코호트 전체 사용자 수) * 100
SELECT 
    r.first_date as cohort_date,
    c.users as cohort_size,
    r.day_number,
    r.active_users,
    ROUND(100.0 * r.active_users / c.users, 2) as retention_rate
FROM retention_data r
JOIN cohort_size c ON r.first_date = c.first_date
WHERE r.day_number >= 0
ORDER BY r.first_date, r.day_number;
