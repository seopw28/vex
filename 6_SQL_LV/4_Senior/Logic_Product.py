#%%
# Python code to create sample data and use SQL
import sqlite3
import pandas as pd

# Create a connection to a new SQLite database
conn = sqlite3.connect('sample.db')

#%% Complex SQL Test Case: E-commerce Analytics
# Create sample order data
orders_data = {
    'order_id': range(1, 11),
    'customer_id': [1, 2, 1, 3, 2, 4, 3, 1, 2, 4],
    'order_date': ['2023-01-01', '2023-01-02', '2023-02-01', '2023-02-05', 
                   '2023-03-01', '2023-03-15', '2023-04-01', '2023-04-10',
                   '2023-05-01', '2023-05-15'],
    'total_amount': [100, 150, 200, 120, 300, 250, 180, 220, 400, 150]
}
orders_df = pd.DataFrame(orders_data)
orders_df['order_date'] = pd.to_datetime(orders_df['order_date'])
orders_df.to_sql('orders', conn, if_exists='replace', index=False)

# Create sample product data
products_data = {
    'product_id': range(1, 6),
    'product_name': ['Laptop', 'Phone', 'Tablet', 'Watch', 'Headphones'],
    'category': ['Electronics', 'Electronics', 'Electronics', 'Accessories', 'Accessories'],
    'base_price': [1000, 800, 500, 300, 200]
}
products_df = pd.DataFrame(products_data)
products_df.to_sql('products', conn, if_exists='replace', index=False)

# Create order details
order_details_data = {
    'order_id': [1,1,2,3,3,4,5,6,7,8,9,10],
    'product_id': [1,3,2,1,4,5,2,3,1,2,4,5],
    'quantity': [1,2,1,1,1,2,1,1,1,1,2,1],
    'unit_price': [950,480,800,950,300,190,780,490,950,780,290,190]
}
order_details_df = pd.DataFrame(order_details_data)
order_details_df.to_sql('order_details', conn, if_exists='replace', index=False)

# Complex SQL Query Challenge
query = '''
WITH customer_stats AS (
    SELECT 
        o.customer_id,
        COUNT(DISTINCT o.order_id) as total_orders,
        SUM(od.quantity * od.unit_price) as total_spent,
        AVG(od.unit_price - p.base_price) as avg_discount
    FROM orders o
    JOIN order_details od ON o.order_id = od.order_id
    JOIN products p ON od.product_id = p.product_id
    GROUP BY o.customer_id
),
monthly_trends AS (
    SELECT 
        strftime('%Y-%m', o.order_date) as month,
        p.category,
        SUM(od.quantity) as total_quantity,
        SUM(od.quantity * od.unit_price) as revenue,
        LAG(SUM(od.quantity * od.unit_price)) OVER (
            PARTITION BY p.category 
            ORDER BY strftime('%Y-%m', o.order_date)
        ) as prev_month_revenue
    FROM orders o
    JOIN order_details od ON o.order_id = od.order_id
    JOIN products p ON od.product_id = p.product_id
    GROUP BY strftime('%Y-%m', o.order_date), p.category
)
SELECT 
    mt.month,
    mt.category,
    mt.revenue,
    mt.prev_month_revenue,
    ROUND(((mt.revenue - mt.prev_month_revenue) * 100.0 / 
        NULLIF(mt.prev_month_revenue, 0)), 2) as revenue_growth,
    mt.total_quantity,
    ROUND(mt.revenue * 100.0 / SUM(mt.revenue) OVER (
        PARTITION BY mt.month
    ), 2) as category_revenue_share_pct
FROM monthly_trends mt
ORDER BY mt.month, mt.revenue DESC;
'''

print("\nComplex E-commerce Analysis Results:")
print(pd.read_sql_query(query, conn))

# Close connection
conn.close()
# %%
