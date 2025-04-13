#%%
import pandas as pd
import plotly.graph_objects as go
import numpy as np

#%%
def heart_metrics_analysis(df):
    """
    Analyze HEART metrics (Happiness, Engagement, Adoption, Retention, Task Success)
    HEART framework is a user-centered analytics approach developed by Google
    """
    metrics = {
        'Happiness': df['satisfaction_score'].mean(),  # User satisfaction scores (1-5)
        'Engagement': df['session_duration'].mean(),   # Avg session duration in minutes
        'Adoption': len(df[df['is_new_feature_used']])/len(df) * 100,  # % using new features
        'Retention': len(df[df['returned_in_28d']])/len(df) * 100,     # 28-day retention rate
        'Task_Success': df['task_completion_rate'].mean() * 100        # % of successful tasks
    }
    
    # Create radar chart
    categories = list(metrics.keys())
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=[metrics[cat] for cat in categories],
        theta=categories,
        fill='toself',
        name='HEART Metrics'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False,
        title="HEART Metrics Analysis"
    )
    
    return fig, metrics

#%%
def heart_trend_analysis(df):
    """
    Analyze HEART metrics trends over time
    """
    # Group by month and calculate metrics
    monthly_metrics = df.groupby('month').agg({
        'satisfaction_score': 'mean',
        'session_duration': 'mean',
        'is_new_feature_used': lambda x: sum(x)/len(x) * 100,
        'returned_in_28d': lambda x: sum(x)/len(x) * 100,
        'task_completion_rate': lambda x: np.mean(x) * 100
    }).reset_index()
    
    # Create line plot
    fig = go.Figure()
    metrics = ['satisfaction_score', 'session_duration', 'is_new_feature_used', 
              'returned_in_28d', 'task_completion_rate']
    
    for metric in metrics:
        fig.add_trace(go.Scatter(
            x=monthly_metrics['month'],
            y=monthly_metrics[metric],
            name=metric.replace('_', ' ').title()
        ))
    
    fig.update_layout(
        title='HEART Metrics Trends',
        xaxis_title='Month',
        yaxis_title='Score',
        height=600
    )
    
    return fig

#%%
# Create sample data
np.random.seed(42)
months = pd.date_range(start='2023-01-01', end='2023-12-31', freq='M')
n_users = 1000

sample_data = {
    'month': np.repeat(months, n_users),
    'user_id': range(len(months) * n_users),
    'satisfaction_score': np.random.normal(4, 0.5, len(months) * n_users).clip(1, 5),
    'session_duration': np.random.exponential(30, len(months) * n_users),
    'is_new_feature_used': np.random.choice([True, False], len(months) * n_users, p=[0.7, 0.3]),
    'returned_in_28d': np.random.choice([True, False], len(months) * n_users, p=[0.8, 0.2]),
    'task_completion_rate': np.random.beta(8, 2, len(months) * n_users)
}

df = pd.DataFrame(sample_data)

# Generate and display the HEART metrics analysis
metrics_fig, metrics = heart_metrics_analysis(df)
metrics_fig.show()

print("\nHEART Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.1f}")

# Generate and display the trends analysis
trends_fig = heart_trend_analysis(df)
trends_fig.show()

#%%
