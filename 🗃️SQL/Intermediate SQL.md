- Querying databases
- Count and View specified records
- Understand query execution and style
- Filtering 
- Aggregate functions
- Sorting and grouping

**NOTE**: Throughout this `PostgreSQL` is used.
# Selecting Data
<hr>

Database Schema
![[ss_230243.png]]
## Querying a Database

### **COUNT()**
- `COUNT()`
- Counts the number of records with a value in a field

#### Single field
```sql
SELECT COUNT(birthdate) AS count_birthdates
FROM people;
```

#### Multiple field
```sql
SELECT COUNT(name) AS count_names, COUNT(birthdate) AS count_birthdates
FROM people;
```

#### Using `*` with `COUNT()`
- `COUNT(field_name)` counts values in a field
- `COUNT(*)` counts total number of records in a table
- `*` represents all fields
```sql
SELECT COUNT(*) AS total_records
FROM people;
```

#### COUNT() with DISTINCT
- Combine `COUNT()` with `DISTINCT` to count unique values
```sql
SELECT COUNT(DISTINCT birthdate) AS count_distinct_birthdates
FROM people;
```
- `COUNT()` include duplicates
- `DISTINCT` excludes duplicates


## Query Execution
- SQL is not processed in its written order
```sql
-- Order of Execution
SELECT name  -- 3
FROM people  -- 2
LIMIT 10;    -- 1
```
- `LIMIT` limits how many results we return 
- Good to know processing order for debugging and aliasing
- Aliases are declared in the `SELECT` statement

### Errors
Most common errors:
- Misspelling 
- Incorrect capitalization
- Incorrect or missing punctuation, especially commas

## SQL formatting
Holywell's style guide: https://www.sqlstyle.guide/

# Filtering Records
<hr>

`WHERE` is used to filter the records.

## Filtering numbers
```sql
SELECT title
FROM files
WHERE release_year > 1960;
```

This is the sql standard symbol that means "not equals to" `<>`
```sql
SELECT title
FROM files
WHERE release_year <> 1960;
```

WHERE can also be used with strings:
```sql
SELECT title
FROM films
WHERE country = 'Japan';
```

Order of execution
```sql
SELECT item              -- 3 
FROM coats               -- 1
WHERE color = 'green'    -- 2
LIMIT 5;                 -- 4 
```

## Multiple criteria 
`OR`, `AND`, `BETWEEN` Is used to filter on the multiple criteria
```sql
SELECT *
FROM coats
WHERE color = 'yellow' OR length = 'short';
```

```sql
SELECT *
FROM coats
WHERE color = 'yellow' AND length = 'short';
```

```sql
SELECT *
FROM coats
WHERE buttons BETWEEN 1 AND 5;
```

### OR
- Use when need to satisfy at least one condition
```sql
SELECT title
FROM films
WHERE releas_year = 1994
	OR release_year = 2000;
```

### AND
- Use `AND` if we need to satisfy all criteria
```sql
SELECT title
FROM films
WHERE release_year > 1994
	AND release_year < 2000;
```

### AND, OR
- Filter films releases in 1994 or 1995, and certified PG or R
- Enclose individual clauses in parenthesis 
```sql
SELECT title
FROM files 
WHERE (release_year = 1994 OR release_year = 1995)
	AND (certification = 'PG' OR certification = 'R');
```

### BETWEEN, AND
```sql
SELECT title
FROM films
WHERE release_year >= 1994
	AND release_year <= 2000;
```

is same as 

```sql
SELECT title
FROM films
WHERE release_year
	BETWEEN 1994 AND 2000;
```

### BETWEEN, AND, OR
```sql
SELECT title
FROM films
WHERE release_year
BETWEEN 1994 AND 2000 AND country = 'UK';
```

## Filtering text
- Filter a pattern rather than specific text 
- `LIKE` , `NOT LIKE`, `IN`

### LIKE
- Used to search for a pattern in a field
- `%` match zero, one, or many characters
- `_` match a single character

```sql
SELECT name
FROM people
WHERE name LIKE 'Ade%';
```

```sql
SELECT name
FROM people 
WHERE name LIKE 'Ev_';
```

### NOT  LIKE
```sql
SELECT name
FROM people
WHERE name NOT LIKE 'A.%';
```

### Wildcard position
Second letter is r.
```sql
SELECT name
FROM people
WHERE name LIKE '%r';
```

third letter is t.
```sql
SELECT name
FROM people
WHERE name LIKE '__t%';
```

### IN
Replace multiple OR statements

instead of this
```sql
SELECT title
FROM films
WHERE realease_year = 1920
OR release_year = 1930
OR release_year = 1940;
```

use this
```sql
SELECT title 
FROM films
WHERE release_year IN (1920, 1930, 1940);
```

```sql
SELECT title
FROM films
WHERE country IN ('Germany', 'France');
```

## Null Value
- `COUNT(field_name)` includes only non-missing values
- `COUNT(*)` includes missing values

`null` -> missing or unknown values.
- Missing values:
	- Human error
	- Information not available
	- Unknown

### IS NULL
```sql
SELECT name
FROM people
WHERE birthdate IS NULL;
```

### IS NOT  NULL
```sql
SELECT COUNT(name) AS count_birthdates
FROM people
WHERE birthdate IS NOT NULL;
```

### COUNT()  vs  IS NOT  NULL
```sql
SELECT 
	COUNT(certifiation)
	AS count_certification
FROM films; 
```

```sql
SELECT 
	COUNT(certification)
	AS count_certification
FROM films
WHERE certification IS NOT NULL;
```

Both will give same result.

### Null put simply
- `NULL` values are missing values
- Very common
- Use `IS NULL` or `IS NOT NULL` to:
	- Identify missing values
	- Select missing values
	- Exclude missing values

# Aggregate Function
<hr>
- Summarises data
- Aggregate function return a single values
- `AVG()`,  `SUM()`,  `MIN()`,  `MAX()`,  `COUNT()`

## Summarising data
```sql
SELECT AVG()
SELECT SUM()
SELECT MIN()
SELECT MAX()
SELECT COUNT()
FROM movies;
```

- Numerical fields only
	- `AVG()`
	- `SUM()`
- Various data types
	- `COUNT()`
	- `MIN()`
	- `MAX()`

## Summarising subsets
Using WHERE with aggregate functions
```sql
SELECT SUM(budget) AS sum_budget
FROM films
WHERE release_year = 2010;
```

```sql
SELECT MIN(budget) AS min_budget
FROM films
WHERE release_year = 2010;
```

#### ROUND()
- Round a number to a specific decimal
- `ROUND(number_to_round, decimal_places)`
```sql
Select ROUND(AVG(budget), 2) AS avg_budget
FROM films
WHERE release_year >= 2010;
```

##### Round to a whole number
```sql
SELECT ROUND(AVG(budget), 0) AS avg_budget
FROM films
WHERE release_year >= 2010;
```

##### Round using negative parameter
```sql
SELECT ROUND(AVG(budget), -5) AS avg_budget
FROM films
WHERE release_year >= 2010;
```
-  The function is rounding to the left of the decimal point instead of the right
- ROUND() is only used with Numerical Fields only.

## Aliasing and arithmetic

### Arithmetic
`+`, `-`, `*` and `/`![[ss_121537.png]]

### Aggregate functions vs. Arithmetic
![[ss_121704.png]]
- Aggregate functions like `SUM()` performs the calculation vertically
- While the Arithmetic functions adds up to records horizontally

### Aliasing and Arithmetic
- To find the profit made by the films
```sql
SELECT (gross - budget) AS profit
FROM films;
```

### Aliasing and Functions
- When using multiple same function use aliasing
```sql
SELECT MAX(budget) AS max_budget
	MAX(duration) AS max_duration
FROM films;
```

### Order of Execution
1. `FROM`
2. `WHERE`
3. `SELECT` (aliases are defined here)
4. `LIMIT`

- Aliases defined in the `SELECT` clause cannot be used in the `WHERE` clause due to order of execution.
![[ss_122359.png]]

# Sorting and Grouping
<hr>

`ORDER BY`
## Sorting results
- By default sorts in ASCENDING
```sql
SELECT title, budget
FROM films
ORDER BY budget;
```

```SQL
SELECT title, budget
FROM films
ORDER BY title;
```

### ASCending
```sql
SELECT title, budget
FROM films
ORDER BY budget ASC;
```

### DESCending
```sql
SELECT title, budget
FROM films
ORDER BY budget AS DESC;
```

### ORDER BY  Multiple fields
- `ORDER BY field_one, fields_two`
```sql
SELECT title, wins
FROM best_movies
ORDER BY wins DESC;
```

- Think of `fields_two` as a tie-breaker
```sql
SELECT title, wins, imbd_score
FROM best_movies
ORDER BY wins DESC, imbd_score DESC;
```

### Different orders
```sql
SELECT birthdate, name
FROM people
ORDER BY birthdate, name DESC;
```

### Order of Execution
```sql
-- Order of execution
SELECT item              -- 3
FROM coats               -- 1
WHERE color = 'yellow'   -- 2
ORDER BY length          -- 4
LIMIT 3;                 -- 5
```

## Grouping data
`GROUP BY`
- Generally used with Aggregate functions to provide statistics.
```SQL
SELECT certification, COUNT(title) AS title_count
FROM films
GROUP BY certification;
```

### Error Handling
- Error - needed  an aggregate function to show
![[ss_131512.png]]

- Handling - uses count aggregate function around title 
![[ss_131549.png]]

### GROUP BY multiple fields
```sql
SELECT cerification, language, COUNT(title) AS title_count
FROM films
GROUP BY certification, language;
```

### GROUP BY with ORDER BY
```sql
SELECT 
	certification,
	COUNT(title) AS title_count
FROM films
GROUP BY certification;
```

- Order By is always written after Group By
```sql
SELECT
	certification,
	COUNT(title) AS title_count
FROM films
GROUP BY certification
ORDER BY title_count DESC;
```

### Order of Execution
```sql
-- Order of execution
SELECT 
	certification,
	COUNT(title) AS title_count    -- 3
FROM films                         -- 1
GROUP BY certification             -- 2
ORDER BY title_count DESC          -- 4
LIMIT 3;                           -- 5
```

## Filtering grouped data
`HAVING`
 - If we want to filter i grouped data we can't use `WHERE` instead use `HAVING`
```sql
SELECT release_year, COUNT(title) AS title_count
From films
GROUP BY release_year
HAVING COUNT(title) > 10;
```

### Order of Execution
```sql
SELECT certification, COUNT(title) AS title_count  -- 5
FROM films                                         -- 1
WHERE certification IN ('G', 'PG', 'PG-13')        -- 2
GROUP BY certification                             -- 3
HAVING COUNT(title) > 500                          -- 4
ORDER BY title_count DESC                          -- 6
LIMIT 3;                                           -- 7
```

### HAVING  vs  WHERE
- WHERE filters individual records
- HAVING filters grouped records

Q. What films were released in the year 2000?
A. No specific filtering is specified so
```sql
SELECT title
FROM films
WHERE release_year = 2000;
```

Q. In what years was the average film duration over two hours?
A. Here we can see `average` = aggregate function hence using `HAVING`
```sql
SELECT release_year
FROM films
GROUP BY release_year
HAVING AVG(duration) > 120;
```
