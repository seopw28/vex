#%%
import pandas as pd
import plotly.graph_objects as go
import numpy as np

#%%
def aarrr_analysis(df):
    """
    Perform AARRR (Pirate Metrics) analysis on user data
    Acquisition -> Activation -> Retention -> Referral -> Revenue
    """
    # Sample metrics calculation
    metrics = {
        'Acquisition': len(df['user_id'].unique()),  # Unique visitors
        'Activation': len(df[df['signed_up'] == True]),  # Signed up users
        'Retention': len(df[df['returned_in_30d'] == True]),  # 30-day retention
        'Referral': len(df[df['referred_users'] > 0]),  # Users who referred others
        'Revenue': len(df[df['purchase_amount'] > 0])  # Paying users
    }
    
    # Create funnel visualization
    fig = go.Figure(go.Funnel(
        y=list(metrics.keys()),
        x=list(metrics.values()),
        textinfo="value+percent initial"
    ))

    fig.update_layout(
        title="AARRR Funnel Analysis",
        width=800,
        height=500
    )
    
    # Calculate conversion rates
    conversions = {}
    metrics_list = list(metrics.values())
    for i in range(len(metrics_list)-1):
        rate = (metrics_list[i+1] / metrics_list[i]) * 100
        stage = f"{list(metrics.keys())[i]} to {list(metrics.keys())[i+1]}"
        conversions[stage] = f"{rate:.1f}%"
        
    return fig, metrics, conversions

#%%
def cohort_retention_analysis(df):
    """
    Analyze user retention by cohort
    """
    # Group users by cohort (signup month) and calculate retention
    cohort_data = df.groupby(['cohort_month', 'months_since_signup'])['user_id'].count().unstack()
    
    # Calculate retention rates
    retention_rates = cohort_data.div(cohort_data[0], axis=0) * 100
    
    # Heatmap visualization
    fig = go.Figure(data=go.Heatmap(
        z=retention_rates.values,
        x=retention_rates.columns,
        y=retention_rates.index,
        colorscale='RdYlBu',
        text=np.round(retention_rates.values, 1),
        texttemplate='%{text}%'
    ))
    
    fig.update_layout(
        title='Cohort Retention Analysis',
        xaxis_title='Months Since Signup',
        yaxis_title='Cohort Month',
        width=1000,
        height=600
    )
    
    return fig

#%%
# Create sample data for demonstration
sample_data = {
    'user_id': range(1000),
    'signed_up': [True] * 800 + [False] * 200,
    'returned_in_30d': [True] * 600 + [False] * 400,
    'referred_users': [1] * 400 + [0] * 600,
    'purchase_amount': [100] * 300 + [0] * 700,
    'cohort_month': ['2023-01'] * 300 + ['2023-02'] * 300 + ['2023-03'] * 400,
    'months_since_signup': [0] * 400 + [1] * 300 + [2] * 300
}

df = pd.DataFrame(sample_data)

# Generate and display the AARRR funnel analysis
funnel_fig, metrics, conversions = aarrr_analysis(df)
funnel_fig.show()

print("\nConversion Rates:")
for stage, rate in conversions.items():
    print(f"{stage}: {rate}")

# Generate and display the cohort retention analysis
retention_fig = cohort_retention_analysis(df)
retention_fig.show()

# %%
