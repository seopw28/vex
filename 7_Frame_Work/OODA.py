#%%
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

#%%
def generate_sample_data(n_days=90):
    """Generate sample e-commerce app data for OODA loop analysis"""
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
    
    # Base patterns with some randomness
    base = np.random.normal(0, 1, n_days)
    trend = np.linspace(0, 2, n_days)
    seasonality = np.sin(np.linspace(0, 6*np.pi, n_days))
    
    data = {
        'date': dates,
        'daily_active_users': 1000 + 100 * base + 200 * trend + 50 * seasonality,
        'conversion_rate': 0.15 + 0.02 * base + 0.01 * seasonality,
        'cart_abandonment': 0.25 - 0.03 * base + 0.02 * seasonality,
        'avg_session_duration': 300 + 30 * base + 20 * seasonality,
        'app_crashes': 5 + 2 * np.random.poisson(1, n_days),
        'customer_complaints': 10 + 3 * np.random.poisson(1, n_days),
        'revenue': 5000 + 500 * base + 1000 * trend + 200 * seasonality
    }
    
    df = pd.DataFrame(data)
    
    # Ensure non-negative values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].clip(lower=0)
    
    return df

#%%
def observe_phase(df):
    """
    Observe Phase: Monitor key metrics and identify patterns
    """
    # Calculate rolling averages
    metrics = ['daily_active_users', 'conversion_rate', 'cart_abandonment', 
              'avg_session_duration', 'app_crashes', 'customer_complaints', 'revenue']
    
    rolling_df = df[metrics].rolling(window=7).mean()
    
    # Create visualization
    fig = go.Figure()
    
    for metric in metrics:
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=rolling_df[metric],
            name=metric.replace('_', ' ').title(),
            mode='lines'
        ))
    
    fig.update_layout(
        title='Key Metrics Trends (7-day Rolling Average)',
        xaxis_title='Date',
        yaxis_title='Value',
        height=600,
        showlegend=True
    )
    
    return fig, rolling_df

#%%
def orient_phase(df):
    """
    Orient Phase: Analyze relationships and patterns
    """
    # Calculate correlation matrix
    metrics = ['daily_active_users', 'conversion_rate', 'cart_abandonment', 
              'avg_session_duration', 'app_crashes', 'customer_complaints', 'revenue']
    correlation = df[metrics].corr()
    
    # Create correlation heatmap
    fig = go.Figure(data=go.Heatmap(
        z=correlation,
        x=metrics,
        y=metrics,
        colorscale='RdBu',
        text=np.round(correlation, 2),
        texttemplate='%{text}'
    ))
    
    fig.update_layout(
        title='Correlation Analysis of Key Metrics',
        width=800,
        height=800
    )
    
    return fig, correlation

#%%
def decide_phase(df):
    """
    Decide Phase: Segment users and identify areas for improvement
    """
    # Prepare data for clustering
    metrics = ['daily_active_users', 'conversion_rate', 'revenue']
    X = df[metrics].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Create visualization
    fig = go.Figure()
    
    for cluster in range(3):
        cluster_data = df[df['cluster'] == cluster]
        fig.add_trace(go.Scatter3d(
            x=cluster_data['daily_active_users'],
            y=cluster_data['conversion_rate'],
            z=cluster_data['revenue'],
            mode='markers',
            name=f'Cluster {cluster}',
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        title='User Segments Based on Key Metrics',
        scene=dict(
            xaxis_title='Daily Active Users',
            yaxis_title='Conversion Rate',
            zaxis_title='Revenue'
        ),
        height=800
    )
    
    return fig, df['cluster']

#%%
def act_phase(df):
    """
    Act Phase: Generate actionable insights and recommendations
    """
    # Calculate key performance indicators
    kpis = {
        'Revenue Trend': df['revenue'].pct_change().mean() * 100,
        'User Growth': df['daily_active_users'].pct_change().mean() * 100,
        'Avg Conversion Rate': df['conversion_rate'].mean() * 100,
        'Cart Abandonment Rate': df['cart_abandonment'].mean() * 100,
        'App Stability': 1 - (df['app_crashes'].sum() / df['daily_active_users'].sum())
    }
    
    # Create visualization
    fig = go.Figure(data=[
        go.Bar(
            x=list(kpis.keys()),
            y=list(kpis.values()),
            text=np.round(list(kpis.values()), 2),
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title='Key Performance Indicators',
        xaxis_title='Metric',
        yaxis_title='Value',
        height=500
    )
    
    return fig, kpis

#%%
# Generate sample data
df = generate_sample_data()

# Run OODA loop analysis
observe_fig, rolling_metrics = observe_phase(df)
orient_fig, correlation = orient_phase(df)
decide_fig, clusters = decide_phase(df)
act_fig, kpis = act_phase(df)

# Display results
observe_fig.show()
orient_fig.show()
decide_fig.show()
act_fig.show()

print("\nKey Performance Indicators:")
for metric, value in kpis.items():
    print(f"{metric}: {value:.2f}")

#%%
