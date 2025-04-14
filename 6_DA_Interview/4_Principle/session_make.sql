-- 세션 생성 SQL 쿼리
-- 30분(1800초) 이상 차이나는 이벤트를 새로운 세션으로 구분

WITH event_gaps AS (
    
    SELECT 
    
        user_id,
        event_time,
        TIMESTAMPDIFF(SECOND, 
                LAG(event_time) OVER (PARTITION BY user_id ORDER BY event_time),
                event_time) as time_gap

    FROM user_events
),

session_breaks AS (

    SELECT 
        user_id,
        event_time,
        CASE 
            WHEN time_gap IS NULL OR time_gap > 1800 
            THEN 1 
            ELSE 0 
        END as is_new_session
    FROM event_gaps
),

session_numbers AS (

    SELECT 
        user_id,
        event_time,
        SUM(is_new_session) OVER (
            PARTITION BY user_id 
            ORDER BY event_time
            ROWS UNBOUNDED PRECEDING) as session_id

    FROM session_breaks
)

SELECT 
    user_id,
    event_time,
    CONCAT(user_id, '_', session_id) as session_id
FROM session_numbers
ORDER BY user_id, event_time;