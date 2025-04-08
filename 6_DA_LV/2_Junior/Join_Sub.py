#%%
# Python code to create sample data and use SQL
import sqlite3
import pandas as pd
import os

# Create a connection to a new SQLite database
conn = sqlite3.connect('sample.db')

#%% Example 1: Orders and Customers
# Create sample dataframes
customers_data = {
    'customer_id': [1, 2, 3, 4, 5],
    'name': ['John Smith', 'Mary Johnson', 'Robert Lee', 'Sarah Wilson', 'Michael Brown'],
    'age': [35, 28, 42, 31, 45],
    'email': ['john@email.com', 'mary@email.com', 'robert@email.com', 'sarah@email.com', 'michael@email.com']
}

customers_df = pd.DataFrame(customers_data)

orders_data = {
    'order_id': [1, 2, 3, 4, 5, 6],
    'customer_id': [1, 2, 1, 3, 4, 2],
    'amount': [100, 200, 150, 300, 250, 175]
}

orders_df = pd.DataFrame(orders_data)

# Save dataframes to SQL tables
customers_df.to_sql('customers', conn, if_exists='replace', index=False)
orders_df.to_sql('orders', conn, if_exists='replace', index=False)

# Query using JOIN
query1 = '''
SELECT c.name, SUM(o.amount) as total_spent
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.name
HAVING total_spent > 200
'''

print("Customers who spent more than $200:")
print(pd.read_sql_query(query1, conn))

#%% Example 2: Using WITH clause
# Create sample dataframe for products and sales
products_data = {
    'product_id': [1, 2, 3, 4, 5],
    'product_name': ['Laptop', 'Phone', 'Tablet', 'Headphones', 'Mouse'],
    'price': [899, 699, 499, 199, 49],
    'category': ['Electronics', 'Electronics', 'Electronics', 'Accessories', 'Accessories']
}
products_df = pd.DataFrame(products_data)

sales_data = {
    'sale_id': [1, 2, 3, 4, 5, 6],
    'product_id': [1, 2, 1, 3, 4, 5],
    'quantity': [2, 3, 1, 4, 5, 10]
}
sales_df = pd.DataFrame(sales_data)

# Save dataframes to SQL tables
products_df.to_sql('products', conn, if_exists='replace', index=False)
sales_df.to_sql('sales', conn, if_exists='replace', index=False)

# Query using WITH clause
query2 = ''' 
WITH product_sales AS (
    SELECT p.product_name, p.category, SUM(s.quantity) as total_sold
    FROM products p
    JOIN sales s ON p.product_id = s.product_id
    GROUP BY p.product_name, p.category
)
SELECT category, AVG(total_sold) as avg_sales
FROM product_sales
GROUP BY category
--
'''

print("\nAverage sales by category:")
print(pd.read_sql_query(query2, conn))

#%% Example 3: Subquery example
# Create sample dataframe for employees
employees_data = {
    'employee_id': [1, 2, 3, 4, 5],
    'name': ['Alice Cooper', 'Bob Wilson', 'Carol Martinez', 'David Kim', 'Eva Chen'],
    'department': ['Sales', 'IT', 'HR', 'Marketing', 'Finance'],
    'salary': [50000, 65000, 45000, 55000, 60000]
}
employees_df = pd.DataFrame(employees_data)

# Save dataframe to SQL table
employees_df.to_sql('employees', conn, if_exists='replace', index=False)

# Query using subquery
query3 = '''
SELECT name, department, salary
FROM employees
WHERE salary > (
    SELECT AVG(salary)
    FROM employees
)
ORDER BY salary DESC
'''

print("\nEmployees with above-average salary:")
print(pd.read_sql_query(query3, conn))

# %%
