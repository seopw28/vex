-- String Functions Comparison Across Different SQL Engines
-- This file contains examples of common string functions in Hive, Oracle, Presto, Redshift, and Impala

-- 1. String Length and Case Manipulation
-- Hive
SELECT length('Hello World'),                    -- Get string length
       lower('Hello World'),                     -- Convert to lowercase
       upper('hello world'),                     -- Convert to uppercase
       initcap('hello world');                   -- Capitalize first letter of each word

-- Redshift
SELECT LENGTH('Hello World'),
       LOWER('Hello World'),
       UPPER('hello world'),
       INITCAP('hello world');

-- Impala
SELECT length('Hello World'),
       lower('Hello World'),
       upper('hello world'),
       initcap('hello world');

-- Presto
SELECT length('Hello World'),
       lower('Hello World'),
       upper('hello world'),
       initcap('hello world');

-- Oracle
SELECT LENGTH('Hello World'),
       LOWER('Hello World'),
       UPPER('hello world'),
       INITCAP('hello world')
FROM DUAL;

-- 2. String Concatenation
-- Hive
SELECT concat('Hello', ' ', 'World'),
       'Hello' || ' ' || 'World';

-- Redshift
SELECT CONCAT('Hello', ' ', 'World'),
       'Hello' || ' ' || 'World';

-- Impala
SELECT concat('Hello', ' ', 'World'),
       'Hello' || ' ' || 'World';

-- Presto
SELECT concat('Hello', ' ', 'World'),
       'Hello' || ' ' || 'World';

-- Oracle
SELECT CONCAT(CONCAT('Hello', ' '), 'World'),
       'Hello' || ' ' || 'World'
FROM DUAL;

-- 3. String Trimming and Padding
-- Hive
SELECT trim('  Hello World  '),                 -- Remove leading and trailing spaces
       ltrim('  Hello World  '),                -- Remove leading spaces
       rtrim('  Hello World  '),                -- Remove trailing spaces
       lpad('Hello', 10, '*'),                 -- Left pad with *
       rpad('Hello', 10, '*');                 -- Right pad with *

-- Redshift
SELECT TRIM('  Hello World  '),
       LTRIM('  Hello World  '),
       RTRIM('  Hello World  '),
       LPAD('Hello', 10, '*'),
       RPAD('Hello', 10, '*');

-- Impala
SELECT trim('  Hello World  '),
       ltrim('  Hello World  '),
       rtrim('  Hello World  '),
       lpad('Hello', 10, '*'),
       rpad('Hello', 10, '*');

-- Presto
SELECT trim('  Hello World  '),
       ltrim('  Hello World  '),
       rtrim('  Hello World  '),
       lpad('Hello', 10, '*'),
       rpad('Hello', 10, '*');

-- Oracle
SELECT TRIM('  Hello World  '),
       LTRIM('  Hello World  '),
       RTRIM('  Hello World  '),
       LPAD('Hello', 10, '*'),
       RPAD('Hello', 10, '*')
FROM DUAL;

-- 4. String Extraction and Position
-- Hive
SELECT substring('Hello World', 1, 5),          -- Extract substring
       substr('Hello World', 1, 5),             -- Alternative substring
       position('World' IN 'Hello World'),      -- Find position of substring
       locate('World', 'Hello World', 1);       -- Find position with start position

-- Redshift
SELECT SUBSTRING('Hello World', 1, 5),
       SUBSTR('Hello World', 1, 5),
       POSITION('World' IN 'Hello World'),
       STRPOS('Hello World', 'World');

-- Impala
SELECT substring('Hello World', 1, 5),
       substr('Hello World', 1, 5),
       position('World' IN 'Hello World'),
       locate('World', 'Hello World', 1);

-- Presto
SELECT substring('Hello World', 1, 5),
       substr('Hello World', 1, 5),
       position('World' IN 'Hello World'),
       strpos('Hello World', 'World');

-- Oracle
SELECT SUBSTR('Hello World', 1, 5),
       INSTR('Hello World', 'World', 1)
FROM DUAL;

-- 5. String Replacement and Translation
-- Hive
SELECT replace('Hello World', 'World', 'SQL'),  -- Replace substring
       translate('Hello World', 'ol', 'OL');    -- Translate characters

-- Redshift
SELECT REPLACE('Hello World', 'World', 'SQL'),
       TRANSLATE('Hello World', 'ol', 'OL');

-- Impala
SELECT replace('Hello World', 'World', 'SQL'),
       translate('Hello World', 'ol', 'OL');

-- Presto
SELECT replace('Hello World', 'World', 'SQL'),
       translate('Hello World', 'ol', 'OL');

-- Oracle
SELECT REPLACE('Hello World', 'World', 'SQL'),
       TRANSLATE('Hello World', 'ol', 'OL')
FROM DUAL;

-- 6. String Splitting and Joining
-- Hive
SELECT split('Hello,World,SQL', ','),           -- Split string into array
       array_join(split('Hello,World,SQL', ','), ' ');  -- Join array into string

-- Redshift
SELECT SPLIT_PART('Hello,World,SQL', ',', 1),   -- Get part by position
       LISTAGG(column_name, ',') WITHIN GROUP (ORDER BY column_name);  -- Aggregate to string

-- Impala
SELECT split('Hello,World,SQL', ','),
       group_concat(column_name, ',');          -- Join multiple rows

-- Presto
SELECT split('Hello,World,SQL', ','),
       array_join(split('Hello,World,SQL', ','), ' ');

-- Oracle
SELECT REGEXP_SUBSTR('Hello,World,SQL', '[^,]+', 1, 1),  -- Get first part
       LISTAGG(column_name, ',') WITHIN GROUP (ORDER BY column_name)
FROM DUAL;

-- 7. Regular Expressions
-- Hive
SELECT regexp_extract('Hello123World', '(\\d+)', 1),     -- Extract using regex
       regexp_replace('Hello123World', '\\d+', '');      -- Replace using regex

-- Redshift
SELECT REGEXP_SUBSTR('Hello123World', '\\d+'),
       REGEXP_REPLACE('Hello123World', '\\d+', '');

-- Impala
SELECT regexp_extract('Hello123World', '(\\d+)', 1),
       regexp_replace('Hello123World', '\\d+', '');

-- Presto
SELECT regexp_extract('Hello123World', '(\\d+)', 1),
       regexp_replace('Hello123World', '\\d+', '');

-- Oracle
SELECT REGEXP_SUBSTR('Hello123World', '\\d+'),
       REGEXP_REPLACE('Hello123World', '\\d+', '')
FROM DUAL;

-- 8. String Comparison and Pattern Matching
-- Hive
SELECT 'Hello' LIKE 'He%',                     -- Pattern matching
       'Hello' SIMILAR TO 'He%',               -- Similar to pattern
       soundex('Hello'),                       -- Soundex code
       levenshtein('Hello', 'Hallo');          -- String distance

-- Redshift
SELECT 'Hello' LIKE 'He%',
       'Hello' SIMILAR TO 'He%',
       SOUNDEX('Hello');

-- Impala
SELECT 'Hello' LIKE 'He%',
       regexp_like('Hello', '^He.*'),
       soundex('Hello');

-- Presto
SELECT 'Hello' LIKE 'He%',
       regexp_like('Hello', '^He.*'),
       soundex('Hello');

-- Oracle
SELECT 'Hello' LIKE 'He%',
       REGEXP_LIKE('Hello', '^He.*'),
       SOUNDEX('Hello')
FROM DUAL;
