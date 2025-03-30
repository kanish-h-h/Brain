# Introduction
<hr>

- Define relationship between tables of data inside the database.
![[ss_192054.png]]

**Database advantages** 
- More storage than spreadsheets
- storage is more secure
- multiple person can access the data at the same time using queries.
## SQL

- Structured Query Language
- most used programming language for databases.
```sql
SELECT *
FROM patrons
LIMIT 30
```

## Tables

- Table rows and columns are referred to as *records* and *fields*
- Fields are set at database creation ; there is no limit to the number of records.

Tables names should be....
- be lowercase
- have no spaces - use underscores `_` instead
- refer to a collection group or be plural

### **Records**
A record is a row that holds data on an individual observation.
![[ss_193138.png]]

### **Fields**
A field is a column that holds one piece of information about all records.
![[ss_193303.png]]

field names should...
- be lowercase
- have no spaces
- be singular
- be different from other field name 
- be different from the table name

### **Assigned seats**
- *Unique identifiers* are used to identify records in a table
- They are unique and often numbers
![[ss_193708.png]]

The more the table the better
![[ss_193750.png]]

## Data

- Different types of data are stored differently and take up different space
- Some operations only apply to certain data types
![[ss_194012.png]]

### Strings
- A string is a sequence of character such as letters or punctuation
- `VARCHAR` is a flexible and popular string data type in SQL.
![[ss_194307.png]]

### Integers
- Integers store whole numbers
- `INT` is a flexible and popular datatype in SQL
![[ss_194450.png]]

### Floats
- Floats store numbers that include a fractional part
- `NUMERIC` is a flexible and popular float data type in SQL
![[ss_194658.png]]

## Schemas
<hr>

- Blueprint of a database
- Schema shows a database's design  and relationship between tables![[ss_194853.png]]

# Queries
<hr>

## Keywords

*Keywords* are reserved words for operations
Common keywords: `SELECT`, `FROM`
![[ss_202440.png]]

Single field
```sql
SELECT name
FROM patrons;
```

Multiple field
```sql
SELECT name, car_num
FROM patrons;
```

### **Aliasing**
Use *aliasing* to rename columns
```sql
SELECT name AS first_name, year_hired
FROM employees;
```

### **Distinct records**
To get a list of unique combination with no repeat values we use `DISTINCT`
```SQL
SELECT DISTINCT year_hired
FROM employees;
```

can be used with multiple fields 
```SQL
SELECT DISTINCT dept_id, year_hired
FROM employees;
```

### **Views**
- A view is a virtual table that is the result of a saved SQL `SELECT` statement
- It is not stored in a database
- But rather stored for future use
```sql
CREATE VIEW employee_hire_years AS
SELECT id, name, year_hired
FROM employees;
```

now use this view
```sql
SELECT id, name
FROM employee_hire_years;
```

## SQL flavors
<hr>

- Both free and paid
- All used with relational databases
- Vast majority of keywords are the same
- All must follow universal standards
- Only the additions on top of these standards make flavors different

### **Two popular SQL flavors**

#### **PostgreSQL** 
- Free and open source relational database system
- Created at the University of California, Berkeley
- "PostgreSQL" refers to both the PostgreSQL database system and its associated SQL flavor
`QUERIES`
![[ss_210618.png]]

#### **SQL Server**
- Has free and paid versions
- Created by Microsoft
- T-SQL is Microsoft's SQL flavor, used with SQL Server databases
`SERVER`
 ![[ss_210644.png]]
 