#%%
# Python code to create sample data and use SQL
import sqlite3
import pandas as pd

# Create a connection to a new SQLite database
conn = sqlite3.connect('sample.db')
#%% Example 1: Customers
# Create sample dataframes
customers_data = {
    'name': ['John Smith', 'Mary Johnson', 'Robert Lee', 'Sarah Wilson', 'Michael Brown'],
    'age': [35, 28, 42, 31, 45],
    'email': ['john@email.com', 'mary@email.com', 'robert@email.com', 'sarah@email.com', 'michael@email.com']
}
customers_df = pd.DataFrame(customers_data)

# Save dataframe to SQL table
customers_df.to_sql('customers', conn, if_exists='replace', index=False)

# Query customers
query1 = '''

SELECT name, age 
FROM customers 
WHERE age >= 30

'''

print("Customers aged 30 or above:")
print(pd.read_sql_query(query1, conn))

#%% Example 2: Products
# Create sample dataframe
products_data = {
    'product_name': ['Laptop', 'Phone', 'Tablet', 'Headphones', 'Mouse'],
    'price': [899, 699, 499, 199, 49],
    'stock': [10, 15, 20, 30, 50]
}
products_df = pd.DataFrame(products_data)

# Save dataframe to SQL table
products_df.to_sql('products', conn, if_exists='replace', index=False)

# Query products
query2 = ''' 

SELECT product_name, price 
FROM products 
WHERE price <= 1000 
ORDER BY price ASC

'''

print("\nProducts under $1000:")
print(pd.read_sql_query(query2, conn))

#%% Example 3: Employees
# Create sample dataframe
employees_data = {
    'name': ['Alice Cooper', 'Bob Wilson', 'Carol Martinez', 'David Kim', 'Eva Chen'],
    'department': ['Sales', 'IT', 'HR', 'Marketing', 'Finance'],
    'salary': [50000, 65000, 45000, 55000, 60000]
}
employees_df = pd.DataFrame(employees_data)

# Save dataframe to SQL table
employees_df.to_sql('employees', conn, if_exists='replace', index=False)

# Query employees
query3 = '''

SELECT department , count(distinct name) as num_employees
FROM employees
GROUP BY department

'''
print("\nEmployees and their departments:")
print(pd.read_sql_query(query3, conn))

# %%
