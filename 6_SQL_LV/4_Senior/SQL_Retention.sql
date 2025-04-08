WITH user_activity AS (
    -- Get first activity date for each user
    SELECT 
        user_id,
        DATE(MIN(event_time)) as first_date,
        DATE(event_time) as activity_date
    FROM user_events
    GROUP BY user_id, DATE(event_time)
),

cohort_size AS (
    -- Calculate cohort sizes
    SELECT 
        first_date,
        COUNT(DISTINCT user_id) as users
    FROM user_activity
    GROUP BY first_date
),

retention_data AS (
    -- Calculate days between first activity and subsequent activities
    SELECT
        a.first_date,
        a.activity_date,
        COUNT(DISTINCT a.user_id) as active_users,
        DATE_DIFF('day', a.first_date, a.activity_date) as day_number
    FROM user_activity a
    GROUP BY a.first_date, a.activity_date
)

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
