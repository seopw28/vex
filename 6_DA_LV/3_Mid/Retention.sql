WITH user_min_dt AS (
    -- 각 사용자별 첫 활동일과 모든 활동일을 추출
    -- first_dt: 사용자의 최초 활동일
    -- act_dt: 사용자의 모든 활동일
    SELECT 
        user_id,
        DATE(MIN(event_time)) as first_dt,
        DATE(event_time) as act_dt
    FROM user_events
    GROUP BY 1, 3
),

day_diff AS (
    -- 리텐션 데이터 계산
    -- 최초 활동일과 이후 활동들 간의 일수 차이 계산
    -- daily_uv: 각 시점별 활성 사용자 수
    -- dt_diff: 최초 활동일로부터 경과된 일수
    SELECT
        a.first_dt,
        a.act_dt,
        COUNT(DISTINCT a.user_id) as daily_uv,
        DATE_DIFF('day', a.first_dt, a.act_dt) as dt_diff
    FROM user_min_dt a
    GROUP BY 1, 2 -- 4는 1,2 에 의한 고정값을 갖게 됨.
)

first_uv AS (
    -- 코호트 크기 계산
    -- 각 최초 활동일(first_dt)별 순수 사용자 수 집계
    SELECT 
        first_dt,
        COUNT(DISTINCT user_id) as first_uv
    FROM user_min_dt
    GROUP BY 1
),



-- 최종 리텐션 데이터 출력
-- first_dt: 코호트의 시작일
-- first_uv: 해당 코호트의 전체 사용자 수
-- dt_diff: 경과 일수
-- daily_uv: 해당 일의 활성 사용자 수
-- retention_rate: 리텐션율(%) = (활성 사용자 수 / 코호트 전체 사용자 수) * 100
SELECT 
    r.first_dt 
    c.first_uv,
    r.dt_diff,
    r.daily_uv,
    ROUND(100.0 * r.daily_uv / c.first_uv, 2) as retention_rate
FROM day_diff r
JOIN first_uv c ON r.first_dt = c.first_dt
WHERE r.dt_diff >= 0
;

