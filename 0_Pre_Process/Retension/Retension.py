#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

#%%
# Function to generate retention sample data
def generate_retention_data(num_users=1000, days=30):
    """
    Generate sample retention data for a cohort of users over a specified number of days.
    
    Parameters:
    -----------
    num_users : int
        Number of users in the cohort
    days : int
        Number of days to track retention
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing retention data
    """
    # Create a date range for the cohort
    cohort_date = datetime.now() - timedelta(days=days)
    
    # Generate user IDs
    user_ids = [f"user_{i}" for i in range(num_users)]
    
    # Generate signup dates (all users sign up on the same day for a cohort)
    signup_dates = [cohort_date] * num_users
    
    # Generate retention data (whether users returned on each day)
    retention_data = []
    
    for day in range(days + 1):
        # Calculate retention probability (decreases over time)
        base_retention = 0.8  # 80% retention on day 1
        decay_factor = 0.95  # 5% decay per day
        retention_prob = base_retention * (decay_factor ** day)
        
        # Add some random noise
        retention_prob = max(0.1, min(0.99, retention_prob + np.random.normal(0, 0.05)))
        
        # Determine which users returned on this day
        returned = np.random.choice([0, 1], size=num_users, p=[1-retention_prob, retention_prob])
        
        # Add to retention data
        for i, (user_id, signup_date, ret) in enumerate(zip(user_ids, signup_dates, returned)):
            retention_data.append({
                'user_id': user_id,
                'signup_date': signup_date,
                'day': day,
                'returned': ret
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(retention_data)
    return df

#%% Function to create a retention heatmap
def create_retention_heatmap(retention_df, title="User Retention Heatmap"):
    """
    Create a retention heatmap from retention data.
    
    Parameters:
    -----------
    retention_df : pandas.DataFrame
        DataFrame containing retention data
    title : str
        Title for the heatmap
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    # Pivot the data to create a matrix of retention rates
    retention_matrix = retention_df.pivot_table(
        index='signup_date',
        columns='day',
        values='returned',
        aggfunc='mean'
    )
    
    # Convert to percentage
    retention_matrix = retention_matrix * 100
    
    # Create the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        retention_matrix,
        annot=True,
        fmt='.1f',
        cmap='YlGnBu',
        cbar_kws={'label': 'Retention Rate (%)'}
    )
    plt.title(title)
    plt.xlabel('Days Since Signup')
    plt.ylabel('Cohort Date')
    plt.tight_layout()
    
    return plt.gcf()

#%% Function to create a retention curve
# Function to create a retention curve
def create_retention_curve(retention_df, title="User Retention Curve"):
    """
    Create a retention curve from retention data.
    
    Parameters:
    -----------
    retention_df : pandas.DataFrame
        DataFrame containing retention data
    title : str
        Title for the curve
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    # Calculate average retention rate for each day
    retention_by_day = retention_df.groupby('day')['returned'].mean() * 100
    
    # Create the curve
    plt.figure(figsize=(10, 6))
    plt.plot(retention_by_day.index, retention_by_day.values, marker='o', linewidth=2)
    plt.fill_between(retention_by_day.index, retention_by_day.values, alpha=0.3)
    plt.title(title)
    plt.xlabel('Days Since Signup')
    plt.ylabel('Retention Rate (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    return plt.gcf()

#%%
# Generate sample retention data
print("Generating retention sample data...")
retention_data = generate_retention_data(num_users=1000, days=30)

# Save the data to CSV
retention_data.to_csv('retention_sample_data.csv', index=False)
print("Sample data saved to 'retention_sample_data.csv'")

#%% Create and save the retention heatmap
print("Creating retention heatmap...")
heatmap_fig = create_retention_heatmap(retention_data)
heatmap_fig.savefig('retention_heatmap.png')
print("Retention heatmap saved to 'retention_heatmap.png'")

#%% Create and save the retention curve
print("Creating retention curve...")
curve_fig = create_retention_curve(retention_data)
curve_fig.savefig('retention_curve.png')
print("Retention curve saved to 'retention_curve.png'")

#%% Display summary statistics
print("\nRetention Summary:")
retention_by_day = retention_data.groupby('day')['returned'].mean() * 100
print(retention_by_day.round(2))

#%% Main function to run the analysis
def main():
    # Generate sample retention data
    print("Generating retention sample data...")
    retention_data = generate_retention_data(num_users=1000, days=30)
    
    # Save the data to CSV
    retention_data.to_csv('retention_sample_data.csv', index=False)
    print("Sample data saved to 'retention_sample_data.csv'")
    
    # Create and save the retention heatmap
    print("Creating retention heatmap...")
    heatmap_fig = create_retention_heatmap(retention_data)
    heatmap_fig.savefig('retention_heatmap.png')
    print("Retention heatmap saved to 'retention_heatmap.png'")
    
    # Create and save the retention curve
    print("Creating retention curve...")
    curve_fig = create_retention_curve(retention_data)
    curve_fig.savefig('retention_curve.png')
    print("Retention curve saved to 'retention_curve.png'")
    
    # Display summary statistics
    print("\nRetention Summary:")
    retention_by_day = retention_data.groupby('day')['returned'].mean() * 100
    print(retention_by_day.round(2))

if __name__ == "__main__":
    main()

# %%
