-- Date Functions Comparison Across Different SQL Engines
-- This file contains examples of common date functions in Hive, Redshift, Impala, Presto, and Oracle

-- 1. Current Date and Time
-- Hive
SELECT current_date, current_timestamp;

-- Redshift
SELECT CURRENT_DATE, CURRENT_TIMESTAMP;

-- Impala
SELECT CURRENT_DATE(), CURRENT_TIMESTAMP();

-- Presto
SELECT CURRENT_DATE, CURRENT_TIMESTAMP;

-- Oracle
SELECT SYSDATE, CURRENT_TIMESTAMP FROM DUAL;


-- 2. Date Formatting
-- Hive
SELECT date_format(current_timestamp, 'yyyy-MM-dd HH:mm:ss');

-- Redshift
SELECT TO_CHAR(CURRENT_TIMESTAMP, 'YYYY-MM-DD HH24:MI:SS');

-- Impala
SELECT from_unixtime(unix_timestamp(), 'yyyy-MM-dd HH:mm:ss');

-- Presto
SELECT format_datetime(current_timestamp, 'yyyy-MM-dd HH:mm:ss');

-- Oracle
SELECT TO_CHAR(SYSDATE, 'YYYY-MM-DD HH24:MI:SS') FROM DUAL;


-- 3. Date Arithmetic
-- Hive
SELECT date_add(current_date, 7),  -- Add 7 days
       date_sub(current_date, 7);  -- Subtract 7 days

-- Redshift
SELECT DATEADD(day, 7, CURRENT_DATE),
       DATEADD(day, -7, CURRENT_DATE);

-- Impala
SELECT date_add(current_date(), 7),
       date_sub(current_date(), 7);

-- Presto
SELECT date_add('day', 7, current_date),
       date_add('day', -7, current_date);

-- Oracle
SELECT SYSDATE + 7,
       SYSDATE - 7
FROM DUAL;

-- 4. Date Parts Extraction
-- Hive
SELECT year(current_timestamp),
       month(current_timestamp),
       day(current_timestamp),
       hour(current_timestamp),
       minute(current_timestamp),
       second(current_timestamp);

-- Redshift
SELECT EXTRACT(YEAR FROM CURRENT_TIMESTAMP),
       EXTRACT(MONTH FROM CURRENT_TIMESTAMP),
       EXTRACT(DAY FROM CURRENT_TIMESTAMP),
       EXTRACT(HOUR FROM CURRENT_TIMESTAMP),
       EXTRACT(MINUTE FROM CURRENT_TIMESTAMP),
       EXTRACT(SECOND FROM CURRENT_TIMESTAMP);

-- Impala
SELECT year(current_timestamp()),
       month(current_timestamp()),
       day(current_timestamp()),
       hour(current_timestamp()),
       minute(current_timestamp()),
       second(current_timestamp());

-- Presto
SELECT EXTRACT(YEAR FROM current_timestamp),
       EXTRACT(MONTH FROM current_timestamp),
       EXTRACT(DAY FROM current_timestamp),
       EXTRACT(HOUR FROM current_timestamp),
       EXTRACT(MINUTE FROM current_timestamp),
       EXTRACT(SECOND FROM current_timestamp);

-- Oracle
SELECT EXTRACT(YEAR FROM SYSDATE),
       EXTRACT(MONTH FROM SYSDATE),
       EXTRACT(DAY FROM SYSDATE),
       EXTRACT(HOUR FROM CURRENT_TIMESTAMP),
       EXTRACT(MINUTE FROM CURRENT_TIMESTAMP),
       EXTRACT(SECOND FROM CURRENT_TIMESTAMP)
FROM DUAL;

-- 5. Date Difference
-- Hive
SELECT datediff(current_date, date_sub(current_date, 30));

-- Redshift
SELECT DATEDIFF(day, DATEADD(day, -30, CURRENT_DATE), CURRENT_DATE);

-- Impala
SELECT datediff(current_date(), date_sub(current_date(), 30));

-- Presto
SELECT date_diff('day', date_add('day', -30, current_date), current_date);

-- Oracle
SELECT SYSDATE - (SYSDATE - 30)
FROM DUAL;

-- 6. First/Last Day of Month
-- Hive
SELECT trunc(current_date, 'MM'),  -- First day
       last_day(current_date);     -- Last day

-- Redshift
SELECT DATE_TRUNC('month', CURRENT_DATE),
       LAST_DAY(CURRENT_DATE);

-- Impala
SELECT trunc(current_date(), 'MM'),
       last_day(current_date());

-- Presto
SELECT date_trunc('month', current_date),
       last_day_of_month(current_date);

-- Oracle
SELECT TRUNC(SYSDATE, 'MM'),
       LAST_DAY(SYSDATE)
FROM DUAL;

-- 7. Date to String and String to Date
-- Hive
SELECT to_date('2024-03-20'),
       from_unixtime(unix_timestamp('2024-03-20', 'yyyy-MM-dd'));

-- Redshift
SELECT TO_DATE('2024-03-20', 'YYYY-MM-DD'),
       TO_CHAR(TO_DATE('2024-03-20', 'YYYY-MM-DD'), 'YYYY-MM-DD');

-- Impala
SELECT to_date('2024-03-20'),
       from_unixtime(unix_timestamp('2024-03-20', 'yyyy-MM-dd'));

-- Presto
SELECT date '2024-03-20',
       format_datetime(parse_datetime('2024-03-20', 'yyyy-MM-dd'), 'yyyy-MM-dd');

-- Oracle
SELECT TO_DATE('2024-03-20', 'YYYY-MM-DD'),
       TO_CHAR(TO_DATE('2024-03-20', 'YYYY-MM-DD'), 'YYYY-MM-DD')
FROM DUAL;

-- 8. Working with Time Zones
-- Hive (Limited timezone support)
SELECT from_utc_timestamp(current_timestamp, 'America/New_York');

-- Redshift
SELECT CONVERT_TIMEZONE('UTC', 'America/New_York', CURRENT_TIMESTAMP);

-- Impala
SELECT from_utc_timestamp(current_timestamp(), 'America/New_York');

-- Presto
SELECT current_timestamp AT TIME ZONE 'America/New_York';

-- Oracle
SELECT FROM_TZ(CAST(SYSDATE AS TIMESTAMP), 'UTC') AT TIME ZONE 'America/New_York'
FROM DUAL;
