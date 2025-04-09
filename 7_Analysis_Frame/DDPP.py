#%%
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#%%
def descriptive_analysis(df):
    """
    Descriptive Analysis: What happened?
    Summarizes key metrics and trends
    """
    # Calculate basic statistics
    stats = df.describe()
    
    # Create time series visualization
    fig = go.Figure()
    
    for metric in ['sales', 'customer_satisfaction', 'marketing_spend']:
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df[metric],
            name=metric.replace('_', ' ').title(),
            mode='lines+markers'
        ))
    
    fig.update_layout(
        title='Key Metrics Over Time',
        xaxis_title='Date',
        yaxis_title='Value',
        height=500
    )
    
    return fig, stats

#%%
def diagnostic_analysis(df):
    """
    Diagnostic Analysis: Why did it happen?
    Analyzes relationships and correlations
    """
    # Calculate correlation matrix
    correlation = df[['sales', 'customer_satisfaction', 'marketing_spend', 
                     'competitor_price', 'seasonality']].corr()
    
    # Create correlation heatmap
    fig = go.Figure(data=go.Heatmap(
        z=correlation,
        x=correlation.columns,
        y=correlation.columns,
        colorscale='RdBu',
        text=np.round(correlation, 2),
        texttemplate='%{text}'
    ))
    
    fig.update_layout(
        title='Correlation Analysis',
        width=800,
        height=800
    )
    
    return fig, correlation

#%%
def predictive_analysis(df):
    """
    Predictive Analysis: What might happen?
    Builds simple prediction model
    """
    # Prepare features and target
    X = df[['marketing_spend', 'competitor_price', 'seasonality']]
    y = df['sales']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Create prediction vs actual plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=y_test,
        y=y_pred,
        mode='markers',
        name='Predictions'
    ))
    
    fig.add_trace(go.Scatter(
        x=[y_test.min(), y_test.max()],
        y=[y_test.min(), y_test.max()],
        mode='lines',
        name='Perfect Prediction',
        line=dict(dash='dash')
    ))
    
    fig.update_layout(
        title='Predicted vs Actual Sales',
        xaxis_title='Actual Sales',
        yaxis_title='Predicted Sales',
        height=600
    )
    
    # Calculate model performance
    performance = {
        'R2 Score': r2_score(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
    }
    
    return fig, performance, model

#%%
def prescriptive_analysis(df, model):
    """
    Prescriptive Analysis: What should we do?
    Simulates different scenarios
    """
    # Create scenarios with different marketing spend
    scenarios = pd.DataFrame({
        'marketing_spend': np.linspace(df['marketing_spend'].min(), 
                                     df['marketing_spend'].max()*1.5, 100),
        'competitor_price': [df['competitor_price'].mean()] * 100,
        'seasonality': [df['seasonality'].mean()] * 100
    })
    
    # Predict outcomes
    predicted_sales = model.predict(scenarios)
    
    # Calculate ROI (simplified)
    scenarios['predicted_sales'] = predicted_sales
    scenarios['roi'] = (scenarios['predicted_sales'] - scenarios['marketing_spend']) / scenarios['marketing_spend']
    
    # Create visualization
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=scenarios['marketing_spend'],
        y=scenarios['predicted_sales'],
        name='Predicted Sales'
    ))
    
    fig.add_trace(go.Scatter(
        x=scenarios['marketing_spend'],
        y=scenarios['roi'],
        name='ROI',
        yaxis='y2'
    ))
    
    fig.update_layout(
        title='Marketing Spend Optimization',
        xaxis_title='Marketing Spend',
        yaxis_title='Predicted Sales',
        yaxis2=dict(
            title='ROI',
            overlaying='y',
            side='right'
        ),
        height=600
    )
    
    # Find optimal point
    optimal_scenario = scenarios.loc[scenarios['roi'].idxmax()]
    
    return fig, optimal_scenario

#%%
# Create sample data
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
n_days = len(dates)

# Generate correlated data
base = np.random.normal(0, 1, n_days)
seasonality = np.sin(np.linspace(0, 4*np.pi, n_days)) + 1

sample_data = {
    'date': dates,
    'sales': 1000 + 100 * base + 200 * seasonality + np.random.normal(0, 50, n_days),
    'customer_satisfaction': 4 + 0.3 * base + np.random.normal(0, 0.2, n_days),
    'marketing_spend': 500 + 50 * base + np.random.normal(0, 30, n_days),
    'competitor_price': 90 + 5 * np.sin(np.linspace(0, 2*np.pi, n_days)) + np.random.normal(0, 2, n_days),
    'seasonality': seasonality
}

df = pd.DataFrame(sample_data)

# Run analyses
desc_fig, stats = descriptive_analysis(df)
diag_fig, correlation = diagnostic_analysis(df)
pred_fig, performance, model = predictive_analysis(df)
presc_fig, optimal_scenario = prescriptive_analysis(df, model)

# Display results
desc_fig.show()
print("\nDescriptive Statistics:")
print(stats)

diag_fig.show()
print("\nCorrelation Analysis:")
print(correlation)

pred_fig.show()
print("\nModel Performance:")
for metric, value in performance.items():
    print(f"{metric}: {value:.3f}")

presc_fig.show()
print("\nOptimal Scenario:")
print(optimal_scenario)

#%%
