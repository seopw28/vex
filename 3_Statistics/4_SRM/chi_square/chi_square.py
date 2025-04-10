#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import os
from fpdf import FPDF
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)

# Function to generate sample data for chi-square analysis
def generate_sample_data(n_samples=1000):
    """
    Generate sample data for chi-square analysis.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
        
    Returns:
    --------
    data : pandas.DataFrame
        DataFrame containing the sample data
    """
    # Define possible values for each category
    sources = ['Website', 'Mobile App', 'Social Media', 'Email', 'Referral']
    devices = ['Desktop', 'Mobile', 'Tablet']
    outcomes = ['Purchase', 'Add to Cart', 'Browse', 'Bounce']
    
    # Generate random data
    data = pd.DataFrame({
        'source': np.random.choice(sources, n_samples, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
        'device': np.random.choice(devices, n_samples, p=[0.5, 0.4, 0.1]),
        'outcome': np.random.choice(outcomes, n_samples, p=[0.2, 0.3, 0.3, 0.2])
    })
    
    # Add some correlations to make the data more realistic
    # For example, mobile devices are more likely to come from mobile app
    mobile_indices = data[data['device'] == 'Mobile'].index
    data.loc[mobile_indices, 'source'] = np.random.choice(
        sources, 
        len(mobile_indices), 
        p=[0.1, 0.5, 0.2, 0.1, 0.1]
    )
    
    # Purchases are more likely from desktop and website
    purchase_indices = data[data['outcome'] == 'Purchase'].index
    data.loc[purchase_indices, 'device'] = np.random.choice(
        devices, 
        len(purchase_indices), 
        p=[0.6, 0.3, 0.1]
    )
    data.loc[purchase_indices, 'source'] = np.random.choice(
        sources, 
        len(purchase_indices), 
        p=[0.4, 0.1, 0.2, 0.2, 0.1]
    )
    
    return data

# Function to perform chi-square test and create visualization
def chi_square_analysis(data, var1, var2, output_path=None):
    """
    Perform chi-square test and create visualization for two categorical variables.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing the data
    var1 : str
        First categorical variable
    var2 : str
        Second categorical variable
    output_path : str, optional
        Path to save the visualization
        
    Returns:
    --------
    dict
        Dictionary containing test results and visualization
    """
    # Create contingency table
    contingency_table = pd.crosstab(data[var1], data[var2])
    
    # Perform chi-square test
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    # Calculate standardized residuals
    standardized_residuals = (contingency_table - expected) / np.sqrt(expected)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Contingency table heatmap
    plt.subplot(2, 2, 1)
    sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlGnBu')
    plt.title(f'Contingency Table: {var1} vs {var2}')
    
    # Plot 2: Expected frequencies heatmap
    plt.subplot(2, 2, 2)
    sns.heatmap(expected, annot=True, fmt='.1f', cmap='YlGnBu')
    plt.title('Expected Frequencies')
    
    # Plot 3: Standardized residuals heatmap
    plt.subplot(2, 2, 3)
    sns.heatmap(standardized_residuals, annot=True, fmt='.2f', cmap='RdBu_r', center=0)
    plt.title('Standardized Residuals')
    
    # Plot 4: Bar chart of observed vs expected
    plt.subplot(2, 2, 4)
    contingency_table_melted = contingency_table.reset_index().melt(id_vars=var1, var_name=var2, value_name='observed')
    expected_melted = pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns).reset_index().melt(id_vars=var1, var_name=var2, value_name='expected')
    
    # Merge observed and expected
    comparison = pd.merge(contingency_table_melted, expected_melted, on=[var1, var2])
    
    # Create grouped bar chart
    comparison_melted = comparison.melt(id_vars=[var1, var2], var_name='type', value_name='count')
    sns.barplot(x=var1, y='count', hue=var2, data=comparison_melted, palette='Set3')
    plt.title(f'Observed vs Expected: {var1} by {var2}')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save visualization if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    
    # Return results
    return {
        'contingency_table': contingency_table,
        'expected': expected,
        'standardized_residuals': standardized_residuals,
        'chi2': chi2,
        'p_value': p_value,
        'dof': dof,
        'visualization': plt.gcf()
    }

# Function to create a PDF report with chi-square analysis results
def create_chi_square_report(data, output_dir):
    """
    Create a PDF report with chi-square analysis results.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing the data
    output_dir : str
        Directory to save the report and visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Perform chi-square tests for different variable pairs
    variable_pairs = [
        ('source', 'device'),
        ('source', 'outcome'),
        ('device', 'outcome')
    ]
    
    results = {}
    for var1, var2 in variable_pairs:
        output_path = os.path.join(output_dir, f"chi_square_{var1}_{var2}.png")
        results[(var1, var2)] = chi_square_analysis(data, var1, var2, output_path)
    
    # Create PDF report
    class PDF(FPDF):
        def header(self):
            # Arial bold 15
            self.set_font('Arial', 'B', 15)
            # Title
            self.cell(0, 10, 'Chi-Square Analysis Report', 0, 1, 'C')
            # Line break
            self.ln(10)
        
        def footer(self):
            # Position at 1.5 cm from bottom
            self.set_y(-15)
            # Arial italic 8
            self.set_font('Arial', 'I', 8)
            # Page number
            self.cell(0, 10, f'Page {self.page_no()}/{{{{nb}}}}', 0, 0, 'C')
        
        def chapter_title(self, title):
            self.set_font('Arial', 'B', 12)
            self.set_fill_color(200, 220, 255)
            self.cell(0, 6, title, 0, 1, 'L', 1)
            self.ln(4)
        
        def chapter_body(self, body):
            self.set_font('Arial', '', 11)
            self.multi_cell(0, 5, body)
            self.ln()
    
    # Create PDF
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # Title page
    pdf.set_font('Arial', 'B', 24)
    pdf.cell(0, 60, 'Chi-Square Analysis', 0, 1, 'C')
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
    pdf.ln(20)
    pdf.cell(0, 10, 'This report contains chi-square analysis of categorical variables', 0, 1, 'C')
    pdf.cell(0, 10, 'in the customer journey data.', 0, 1, 'C')
    
    # Add a new page
    pdf.add_page()
    
    # Introduction
    pdf.chapter_title('Introduction')
    pdf.chapter_body('This report presents chi-square analysis of categorical variables in the customer journey data. '
                    'Chi-square tests are used to determine if there is a significant association between categorical variables. '
                    'The analysis includes contingency tables, expected frequencies, standardized residuals, and visualizations.')
    
    # Data Overview
    pdf.chapter_title('Data Overview')
    pdf.chapter_body(f'The dataset contains {len(data)} customer journey records. '
                    f'Each record represents a customer interaction with the following categorical attributes: '
                    f'source, device, and outcome.')
    
    # Variable Distributions
    pdf.chapter_title('Variable Distributions')
    
    # Source distribution
    source_dist = data['source'].value_counts(normalize=True).round(3) * 100
    source_text = 'Distribution of customer sources:\n\n'
    for source, percentage in source_dist.items():
        source_text += f'{source}: {percentage:.1f}%\n'
    pdf.chapter_body(source_text)
    
    # Device distribution
    device_dist = data['device'].value_counts(normalize=True).round(3) * 100
    device_text = 'Distribution of customer devices:\n\n'
    for device, percentage in device_dist.items():
        device_text += f'{device}: {percentage:.1f}%\n'
    pdf.chapter_body(device_text)
    
    # Outcome distribution
    outcome_dist = data['outcome'].value_counts(normalize=True).round(3) * 100
    outcome_text = 'Distribution of customer outcomes:\n\n'
    for outcome, percentage in outcome_dist.items():
        outcome_text += f'{outcome}: {percentage:.1f}%\n'
    pdf.chapter_body(outcome_text)
    
    # Chi-Square Analysis Results
    for var1, var2 in variable_pairs:
        pdf.add_page()
        pdf.chapter_title(f'Chi-Square Analysis: {var1} vs {var2}')
        
        # Add visualization
        vis_path = os.path.join(output_dir, f"chi_square_{var1}_{var2}.png")
        if os.path.exists(vis_path):
            pdf.image(vis_path, x=10, y=None, w=190)
            pdf.ln(10)
        
        # Add test results
        result = results[(var1, var2)]
        test_text = f'Chi-square test results:\n\n'
        test_text += f'Chi-square statistic: {result["chi2"]:.2f}\n'
        test_text += f'Degrees of freedom: {result["dof"]}\n'
        test_text += f'P-value: {result["p_value"]:.4f}\n\n'
        
        if result["p_value"] < 0.05:
            test_text += 'Conclusion: There is a significant association between the variables (p < 0.05).\n\n'
        else:
            test_text += 'Conclusion: There is no significant association between the variables (p >= 0.05).\n\n'
        
        test_text += 'Interpretation of standardized residuals:\n'
        test_text += '- Values > 2 indicate cells with more observations than expected\n'
        test_text += '- Values < -2 indicate cells with fewer observations than expected\n'
        test_text += '- Values between -2 and 2 indicate cells with approximately expected observations\n'
        
        pdf.chapter_body(test_text)
    
    # Save the PDF
    pdf_path = os.path.join(output_dir, 'chi_square_analysis_report.pdf')
    pdf.output(pdf_path)
    print(f"PDF report saved to {pdf_path}")

#%%
# Generate sample data
print("Generating sample data for chi-square analysis...")
chi_square_data = generate_sample_data(n_samples=1000)

# Display the first few rows
print("\nFirst few rows of the data:")
display(chi_square_data.head())

#%%
# Perform chi-square analysis for source vs device
print("\nPerforming chi-square analysis for source vs device...")
source_device_result = chi_square_analysis(chi_square_data, 'source', 'device')
print(f"Chi-square statistic: {source_device_result['chi2']:.2f}")
print(f"P-value: {source_device_result['p_value']:.4f}")

#%%
# Perform chi-square analysis for source vs outcome
print("\nPerforming chi-square analysis for source vs outcome...")
source_outcome_result = chi_square_analysis(chi_square_data, 'source', 'outcome')
print(f"Chi-square statistic: {source_outcome_result['chi2']:.2f}")
print(f"P-value: {source_outcome_result['p_value']:.4f}")

#%%
# Perform chi-square analysis for device vs outcome
print("\nPerforming chi-square analysis for device vs outcome...")
device_outcome_result = chi_square_analysis(chi_square_data, 'device', 'outcome')
print(f"Chi-square statistic: {device_outcome_result['chi2']:.2f}")
print(f"P-value: {device_outcome_result['p_value']:.4f}")

#%%
# Create PDF report
print("\nCreating PDF report...")
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, "chi_square_analysis")
create_chi_square_report(chi_square_data, output_dir)

#%%
# Main function to run the analysis
def main():
    # Generate sample data
    print("Generating sample data for chi-square analysis...")
    chi_square_data = generate_sample_data(n_samples=1000)
    
    # Create output directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, "chi_square_analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create PDF report
    print("Creating PDF report...")
    create_chi_square_report(chi_square_data, output_dir)
    
    print("Analysis completed successfully!")

if __name__ == "__main__":
    main() 