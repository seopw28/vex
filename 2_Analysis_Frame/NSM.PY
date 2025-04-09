#%%
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

#%%
def north_star_metric_analysis(df):
    """
    Analyze North Star Metric (NSM) and its key supporting metrics
    """
    # Calculate main NSM - Weekly Active Users in this example
    nsm = df.groupby('week')['user_id'].nunique()
    
    # Calculate supporting metrics
    supporting_metrics = df.groupby('week').agg({
        'session_count': 'sum',  # User engagement
        'feature_adoption': 'mean',  # Feature adoption rate
        'retention_rate': 'mean',  # User retention
        'revenue': 'sum'  # Revenue generation
    })
    
    # Create visualization for NSM trend
    fig_nsm = go.Figure()
    fig_nsm.add_trace(go.Scatter(
        x=nsm.index,
        y=nsm.values,
        mode='lines+markers',
        name='Weekly Active Users'
    ))
    
    fig_nsm.update_layout(
        title='North Star Metric: Weekly Active Users',
        xaxis_title='Week',
        yaxis_title='Number of Active Users',
        height=500
    )
    
    # Create visualization for supporting metrics
    fig_supporting = go.Figure()
    
    for metric in supporting_metrics.columns:
        fig_supporting.add_trace(go.Scatter(
            x=supporting_metrics.index,
            y=supporting_metrics[metric],
            mode='lines+markers',
            name=metric.replace('_', ' ').title()
        ))
    
    fig_supporting.update_layout(
        title='Supporting Metrics Trends',
        xaxis_title='Week',
        yaxis_title='Value',
        height=500
    )
    
    return fig_nsm, fig_supporting, nsm, supporting_metrics

#%%
def nsm_impact_analysis(df):
    """
    Analyze the impact of supporting metrics on NSM
    """
    # Calculate correlation matrix
    correlation_matrix = df.groupby('week').agg({
        'active_users': 'sum',
        'session_count': 'sum',
        'feature_adoption': 'mean',
        'retention_rate': 'mean',
        'revenue': 'sum'
    }).corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='RdBu',
        text=np.round(correlation_matrix, 2),
        texttemplate='%{text}'
    ))
    
    fig.update_layout(
        title='Correlation between NSM and Supporting Metrics',
        width=800,
        height=800
    )
    
    return fig, correlation_matrix

#%%
# Create sample data
np.random.seed(42)

# Generate dates for 12 weeks
weeks = pd.date_range(start='2023-01-01', periods=12, freq='W')
n_users = 1000

# Create sample data with weekly metrics
sample_data = {
    'week': np.repeat(weeks, n_users),
    'user_id': range(len(weeks) * n_users),
    'active_users': np.random.choice([0, 1], len(weeks) * n_users, p=[0.3, 0.7]),
    'session_count': np.random.poisson(5, len(weeks) * n_users),
    'feature_adoption': np.random.beta(8, 2, len(weeks) * n_users),
    'retention_rate': np.random.beta(7, 3, len(weeks) * n_users),
    'revenue': np.random.exponential(100, len(weeks) * n_users)
}

df = pd.DataFrame(sample_data)

# Generate and display the NSM analysis
nsm_fig, supporting_fig, nsm, supporting_metrics = north_star_metric_analysis(df)
nsm_fig.show()
supporting_fig.show()

# Generate and display the impact analysis
impact_fig, correlation_matrix = nsm_impact_analysis(df)
impact_fig.show()

print("\nCorrelation Matrix:")
print(correlation_matrix['active_users'].sort_values(ascending=False))

#%%
