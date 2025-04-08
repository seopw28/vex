#%%
# Python code to create sample data and use SQL
import sqlite3
import pandas as pd
import os

# Create a connection to a new SQLite database
conn = sqlite3.connect('sample.db')

#%% Example 1: Sales Analysis with Window Functions
# Create sample dataframes
sales_data = {
    'sale_id': [1, 2, 3, 4, 5, 6, 7, 8],
    'product': ['Laptop', 'Phone', 'Laptop', 'Tablet', 'Phone', 'Laptop', 'Mouse', 'Keyboard'],
    'region': ['North', 'North', 'South', 'South', 'North', 'South', 'North', 'South'],
    'amount': [1200, 800, 1100, 500, 750, 1300, 50, 100]
}
sales_df = pd.DataFrame(sales_data)

# Save dataframe to SQL table
sales_df.to_sql('sales', conn, if_exists='replace', index=False)

# Query using ROW_NUMBER() window function
query1 = '''
SELECT 
    sale_id,
    product,
    region,
    amount,
    ROW_NUMBER() OVER (PARTITION BY region ORDER BY amount DESC) as rank_in_region
FROM sales
'''

print("Sales ranked within each region:")
print(pd.read_sql_query(query1, conn))

#%% Example 2: Running Totals with Window Functions
# Create sample dataframe for monthly sales
monthly_sales_data = {
    'month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
    'sales': [10000, 12000, 9500, 15000, 11000, 13500]
}
monthly_sales_df = pd.DataFrame(monthly_sales_data)

# Save dataframe to SQL table
monthly_sales_df.to_sql('monthly_sales', conn, if_exists='replace', index=False)

# Query using SUM() OVER window function
query2 = '''
SELECT 
    month,
    sales,
    SUM(sales) OVER (ORDER BY rowid) as running_total,
    AVG(sales) OVER (ORDER BY rowid ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) as moving_avg
FROM monthly_sales
'''

print("\nMonthly sales with running totals and 3-month moving average:")
print(pd.read_sql_query(query2, conn))

#%% Example 3: Employee Salary Analysis
# Create sample dataframe for employees
employees_data = {
    'employee_id': [1, 2, 3, 4, 5, 6],
    'name': ['Alice Cooper', 'Bob Wilson', 'Carol Martinez', 'David Kim', 'Eva Chen', 'Frank Zhang'],
    'department': ['Sales', 'Sales', 'IT', 'IT', 'HR', 'HR'],
    'salary': [50000, 65000, 70000, 55000, 60000, 52000]
}
employees_df = pd.DataFrame(employees_data)

# Save dataframe to SQL table
employees_df.to_sql('employees', conn, if_exists='replace', index=False)

# Query using multiple window functions
query3 = '''
SELECT 
    name,
    department,
    salary,
    AVG(salary) OVER (PARTITION BY department) as dept_avg_salary,
    salary - AVG(salary) OVER (PARTITION BY department) as diff_from_dept_avg,
    RANK() OVER (PARTITION BY department ORDER BY salary DESC) as salary_rank
FROM employees
ORDER BY department, salary_rank
--
'''

print("\nEmployee salary analysis by department:")

print(pd.read_sql_query(query3, conn))

# %%
