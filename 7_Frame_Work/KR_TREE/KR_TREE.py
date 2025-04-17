#%%
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx

#%%
def generate_kpi_sample_data(n_days=90):
    """Generate sample KPI metrics data"""
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
    
    # Base patterns
    base = np.random.normal(0, 1, n_days)
    trend = np.linspace(0, 2, n_days)
    seasonality = np.sin(np.linspace(0, 4*np.pi, n_days))
    
    # Revenue metrics
    revenue = 10000 + 1000 * base + 2000 * trend + 500 * seasonality
    avg_order_value = 100 + 10 * base + 5 * seasonality
    orders = revenue / avg_order_value
    
    # Customer metrics
    conversion_rate = 0.15 + 0.02 * base + 0.01 * seasonality
    visitors = orders / conversion_rate
    new_customers = visitors * (0.3 + 0.05 * base)
    repeat_customers = visitors - new_customers
    
    # Product metrics
    items_per_order = 2.5 + 0.2 * base + 0.1 * seasonality
    product_views = visitors * (3 + 0.5 * base)
    cart_abandonment = 0.25 - 0.03 * base + 0.02 * seasonality
    
    data = {
        'date': dates,
        'revenue': revenue,
        'avg_order_value': avg_order_value,
        'orders': orders,
        'conversion_rate': conversion_rate,
        'visitors': visitors,
        'new_customers': new_customers,
        'repeat_customers': repeat_customers,
        'items_per_order': items_per_order,
        'product_views': product_views,
        'cart_abandonment': cart_abandonment
    }
    
    df = pd.DataFrame(data)
    
    # Ensure non-negative values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].clip(lower=0)
    
    return df

#%%
def create_kpi_tree():
    """Create KPI relationship tree structure"""
    G = nx.DiGraph()
    
    # Add nodes
    nodes = {
        'Revenue': {'level': 0},
        'Orders': {'level': 1},
        'AOV': {'level': 1},
        'Visitors': {'level': 2},
        'Conv Rate': {'level': 2},
        'New Cust': {'level': 3},
        'Repeat Cust': {'level': 3},
        'Items/Order': {'level': 2},
        'Prod Views': {'level': 3},
        'Cart Aband': {'level': 3}
    }
    
    G.add_nodes_from([(node, attr) for node, attr in nodes.items()])
    
    # Add edges
    edges = [
        ('Orders', 'Revenue'),
        ('AOV', 'Revenue'),
        ('Visitors', 'Orders'),
        ('Conv Rate', 'Orders'),
        ('New Cust', 'Visitors'),
        ('Repeat Cust', 'Visitors'),
        ('Items/Order', 'AOV'),
        ('Prod Views', 'Conv Rate'),
        ('Cart Aband', 'Conv Rate')
    ]
    
    G.add_edges_from(edges)
    
    return G

#%%
def visualize_kpi_tree(G, df):
    """Visualize KPI tree with metrics"""
    # Calculate positions using hierarchical layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Create edges trace
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    # Create nodes trace
    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Calculate metric value and trend
        if node == 'Revenue':
            value = df['revenue'].mean()
            trend = df['revenue'].pct_change().mean() * 100
        elif node == 'Orders':
            value = df['orders'].mean()
            trend = df['orders'].pct_change().mean() * 100
        elif node == 'AOV':
            value = df['avg_order_value'].mean()
            trend = df['avg_order_value'].pct_change().mean() * 100
        elif node == 'Visitors':
            value = df['visitors'].mean()
            trend = df['visitors'].pct_change().mean() * 100
        elif node == 'Conv Rate':
            value = df['conversion_rate'].mean() * 100
            trend = df['conversion_rate'].pct_change().mean() * 100
        elif node == 'New Cust':
            value = df['new_customers'].mean()
            trend = df['new_customers'].pct_change().mean() * 100
        elif node == 'Repeat Cust':
            value = df['repeat_customers'].mean()
            trend = df['repeat_customers'].pct_change().mean() * 100
        elif node == 'Items/Order':
            value = df['items_per_order'].mean()
            trend = df['items_per_order'].pct_change().mean() * 100
        elif node == 'Prod Views':
            value = df['product_views'].mean()
            trend = df['product_views'].pct_change().mean() * 100
        elif node == 'Cart Aband':
            value = df['cart_abandonment'].mean() * 100
            trend = df['cart_abandonment'].pct_change().mean() * 100
            
        node_text.append(f"{node}<br>Value: {value:.1f}<br>Trend: {trend:+.1f}%")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="top center",
        marker=dict(
            showscale=True,
            colorscale='YlOrRd',
            size=30,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left'
            )
        )
    )

    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title='KPI Relationship Tree',
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                   )
    
    return fig

#%%
# Generate sample data
df = generate_kpi_sample_data()

# Create and visualize KPI tree
G = create_kpi_tree()
fig = visualize_kpi_tree(G, df)

# Save the figure as a PNG file
fig.write_image("kpi_tree.png", width=1200, height=800, scale=2)

# Display the figure
fig.show()

#%%
