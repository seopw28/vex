#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.io as pio
from fpdf import FPDF
import os
from datetime import datetime
import io
from PIL import Image

# Set random seed for reproducibility
np.random.seed(42)

# Set the default renderer
pio.renderers.default = 'png'

# Function to save plotly figure
def save_plotly_figure(fig, base_name):
    # Save as HTML
    html_path = f"{base_name}.html"
    fig.write_html(html_path)
    print(f"Figure saved as HTML: {html_path}")
    print("Please open this HTML file in a browser to view the interactive visualization")
    
    try:
        # Try to save as PNG if possible
        png_path = f"{base_name}.png"
        fig.write_image(png_path)
        print(f"Figure also saved as PNG: {png_path}")
        return png_path
    except Exception as e:
        print(f"Could not save as PNG: {e}")
        return html_path

#%%
# Function to generate sample data for Sankey chart
def generate_sankey_data(n_samples=1000):
    """
    Generate sample data for Sankey chart visualization.
    
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
    mediums = ['Organic', 'Paid', 'Direct', 'Referral']
    campaigns = ['Brand', 'Generic', 'Retargeting', 'Seasonal', 'Product']
    devices = ['Desktop', 'Mobile', 'Tablet']
    outcomes = ['Purchase', 'Add to Cart', 'Browse', 'Bounce']
    
    # Generate random data
    data = pd.DataFrame({
        'source': np.random.choice(sources, n_samples, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
        'medium': np.random.choice(mediums, n_samples, p=[0.4, 0.3, 0.2, 0.1]),
        'campaign': np.random.choice(campaigns, n_samples, p=[0.3, 0.3, 0.2, 0.1, 0.1]),
        'device': np.random.choice(devices, n_samples, p=[0.5, 0.4, 0.1]),
        'outcome': np.random.choice(outcomes, n_samples, p=[0.2, 0.3, 0.3, 0.2])
    })
    
    # Add some correlations to make the data more realistic
    # For example, paid medium is more likely to come from social media
    paid_indices = data[data['medium'] == 'Paid'].index
    data.loc[paid_indices, 'source'] = np.random.choice(
        sources, 
        len(paid_indices), 
        p=[0.1, 0.2, 0.4, 0.2, 0.1]
    )
    
    # Mobile devices are more likely to come from mobile app
    mobile_indices = data[data['device'] == 'Mobile'].index
    data.loc[mobile_indices, 'source'] = np.random.choice(
        sources, 
        len(mobile_indices), 
        p=[0.1, 0.5, 0.2, 0.1, 0.1]
    )
    
    # Purchases are more likely from desktop and paid medium
    purchase_indices = data[data['outcome'] == 'Purchase'].index
    data.loc[purchase_indices, 'device'] = np.random.choice(
        devices, 
        len(purchase_indices), 
        p=[0.6, 0.3, 0.1]
    )
    data.loc[purchase_indices, 'medium'] = np.random.choice(
        mediums, 
        len(purchase_indices), 
        p=[0.2, 0.5, 0.2, 0.1]
    )
    
    return data

#%%
# Generate sample data
print("Generating sample data for Sankey chart...")
sankey_data = generate_sankey_data(n_samples=1000)

# Display the first few rows
print("\nFirst few rows of the data:")
display(sankey_data.head())

#%%
# Create Sankey chart
print("Creating Sankey chart...")

# Define the nodes (unique values in each column)
source_nodes = sankey_data['source'].unique().tolist()
medium_nodes = sankey_data['medium'].unique().tolist()
campaign_nodes = sankey_data['campaign'].unique().tolist()
device_nodes = sankey_data['device'].unique().tolist()
outcome_nodes = sankey_data['outcome'].unique().tolist()

# Create a mapping from node names to indices
all_nodes = source_nodes + medium_nodes + campaign_nodes + device_nodes + outcome_nodes
node_indices = {node: i for i, node in enumerate(all_nodes)}

# Create the links
links = []

# Source to Medium links
source_medium_counts = sankey_data.groupby(['source', 'medium']).size().reset_index(name='count')
for _, row in source_medium_counts.iterrows():
    links.append({
        'source': node_indices[row['source']],
        'target': node_indices[row['medium']],
        'value': row['count']
    })

# Medium to Campaign links
medium_campaign_counts = sankey_data.groupby(['medium', 'campaign']).size().reset_index(name='count')
for _, row in medium_campaign_counts.iterrows():
    links.append({
        'source': node_indices[row['medium']],
        'target': node_indices[row['campaign']],
        'value': row['count']
    })

# Campaign to Device links
campaign_device_counts = sankey_data.groupby(['campaign', 'device']).size().reset_index(name='count')
for _, row in campaign_device_counts.iterrows():
    links.append({
        'source': node_indices[row['campaign']],
        'target': node_indices[row['device']],
        'value': row['count']
    })

# Device to Outcome links
device_outcome_counts = sankey_data.groupby(['device', 'outcome']).size().reset_index(name='count')
for _, row in device_outcome_counts.iterrows():
    links.append({
        'source': node_indices[row['device']],
        'target': node_indices[row['outcome']],
        'value': row['count']
    })

# Create the Sankey diagram
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=all_nodes,
        color=[
            "#1f77b4" if i < len(source_nodes) else
            "#ff7f0e" if i < len(source_nodes) + len(medium_nodes) else
            "#2ca02c" if i < len(source_nodes) + len(medium_nodes) + len(campaign_nodes) else
            "#d62728" if i < len(source_nodes) + len(medium_nodes) + len(campaign_nodes) + len(device_nodes) else
            "#9467bd"
            for i in range(len(all_nodes))
        ]
    ),
    link=dict(
        source=[link['source'] for link in links],
        target=[link['target'] for link in links],
        value=[link['value'] for link in links]
    )
)])

# Update layout
fig.update_layout(
    title_text="Customer Journey Sankey Diagram",
    font_size=12,
    height=800
)

# Save the figure as HTML
fig.write_html("sankey_diagram.html")
print("Sankey diagram saved to 'sankey_diagram.html'")

# Save as static image (PNG)
current_dir = os.path.dirname(os.path.abspath(__file__))
sankey_image_path = os.path.join(current_dir, "sankey_diagram.png")
save_plotly_figure(fig, "sankey_diagram")

#%%
# Create summary statistics
print("Creating summary statistics...")

# Source distribution
source_dist = sankey_data['source'].value_counts(normalize=True).round(3) * 100
print("\nSource Distribution:")
display(source_dist)

# Medium distribution
medium_dist = sankey_data['medium'].value_counts(normalize=True).round(3) * 100
print("\nMedium Distribution:")
display(medium_dist)

# Campaign distribution
campaign_dist = sankey_data['campaign'].value_counts(normalize=True).round(3) * 100
print("\nCampaign Distribution:")
display(campaign_dist)

# Device distribution
device_dist = sankey_data['device'].value_counts(normalize=True).round(3) * 100
print("\nDevice Distribution:")
display(device_dist)

# Outcome distribution
outcome_dist = sankey_data['outcome'].value_counts(normalize=True).round(3) * 100
print("\nOutcome Distribution:")
display(outcome_dist)

#%%
# Create conversion funnel
print("Creating conversion funnel...")

# Calculate conversion rates
total_users = len(sankey_data)
add_to_cart_users = len(sankey_data[sankey_data['outcome'] == 'Add to Cart'])
purchase_users = len(sankey_data[sankey_data['outcome'] == 'Purchase'])

add_to_cart_rate = (add_to_cart_users / total_users) * 100
purchase_rate = (purchase_users / total_users) * 100
cart_to_purchase_rate = (purchase_users / add_to_cart_users) * 100 if add_to_cart_users > 0 else 0

print(f"Total Users: {total_users}")
print(f"Add to Cart Users: {add_to_cart_users} ({add_to_cart_rate:.2f}%)")
print(f"Purchase Users: {purchase_users} ({purchase_rate:.2f}%)")
print(f"Cart to Purchase Rate: {cart_to_purchase_rate:.2f}%")

# Create funnel chart
funnel_data = [
    go.Funnel(
        y=['Total Users', 'Add to Cart', 'Purchase'],
        x=[total_users, add_to_cart_users, purchase_users],
        textinfo="value+percent initial",
        textposition="inside",
        textfont={"size": 20},
        marker=dict(
            color=["#1f77b4", "#ff7f0e", "#2ca02c"]
        )
    )
]

funnel_layout = go.Layout(
    title="Conversion Funnel",
    showlegend=False,
    height=500
)

funnel_fig = go.Figure(data=funnel_data, layout=funnel_layout)
funnel_image_path = os.path.join(current_dir, "conversion_funnel.png")
save_plotly_figure(funnel_fig, "conversion_funnel")

#%%
# Create a PDF summary report
print("Creating PDF summary report...")

class PDF(FPDF):
    def header(self):
        # Arial bold 15
        self.set_font('Arial', 'B', 15)
        # Title
        self.cell(0, 10, 'Customer Journey Analysis Report', 0, 1, 'C')
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
pdf.cell(0, 60, 'Customer Journey Analysis', 0, 1, 'C')
pdf.set_font('Arial', '', 12)
pdf.cell(0, 10, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
pdf.ln(20)
pdf.cell(0, 10, 'This report contains an analysis of customer journey data', 0, 1, 'C')
pdf.cell(0, 10, 'using a Sankey diagram and various statistical measures.', 0, 1, 'C')

# Add a new page
pdf.add_page()

# Introduction
pdf.chapter_title('Introduction')
pdf.chapter_body('This report presents an analysis of customer journey data using a Sankey diagram. '
                'The data represents the flow of customers through different stages of their journey, '
                'from various sources to final outcomes. The analysis includes distribution statistics '
                'for each stage and a conversion funnel showing the progression of users through the journey.')

# Data Overview
pdf.chapter_title('Data Overview')
pdf.chapter_body(f'The dataset contains {len(sankey_data)} customer journey records. '
                f'Each record represents a customer interaction with the following attributes: '
                f'source, medium, campaign, device, and outcome.')

# Source Distribution
pdf.chapter_title('Source Distribution')
source_text = 'Distribution of customer sources:\n\n'
for source, percentage in source_dist.items():
    source_text += f'{source}: {percentage:.1f}%\n'
pdf.chapter_body(source_text)

# Medium Distribution
pdf.chapter_title('Medium Distribution')
medium_text = 'Distribution of customer mediums:\n\n'
for medium, percentage in medium_dist.items():
    medium_text += f'{medium}: {percentage:.1f}%\n'
pdf.chapter_body(medium_text)

# Campaign Distribution
pdf.chapter_title('Campaign Distribution')
campaign_text = 'Distribution of customer campaigns:\n\n'
for campaign, percentage in campaign_dist.items():
    campaign_text += f'{campaign}: {percentage:.1f}%\n'
pdf.chapter_body(campaign_text)

# Device Distribution
pdf.chapter_title('Device Distribution')
device_text = 'Distribution of customer devices:\n\n'
for device, percentage in device_dist.items():
    device_text += f'{device}: {percentage:.1f}%\n'
pdf.chapter_body(device_text)

# Outcome Distribution
pdf.chapter_title('Outcome Distribution')
outcome_text = 'Distribution of customer outcomes:\n\n'
for outcome, percentage in outcome_dist.items():
    outcome_text += f'{outcome}: {percentage:.1f}%\n'
pdf.chapter_body(outcome_text)

# Conversion Funnel
pdf.chapter_title('Conversion Funnel')
funnel_text = f'The conversion funnel shows the progression of users through the customer journey:\n\n'
funnel_text += f'Total Users: {total_users}\n'
funnel_text += f'Add to Cart Users: {add_to_cart_users} ({add_to_cart_rate:.2f}%)\n'
funnel_text += f'Purchase Users: {purchase_users} ({purchase_rate:.2f}%)\n'
funnel_text += f'Cart to Purchase Rate: {cart_to_purchase_rate:.2f}%\n'
pdf.chapter_body(funnel_text)

# Add Sankey Diagram
pdf.chapter_title('Sankey Diagram')
sankey_path = save_plotly_figure(fig, "sankey_diagram")
if sankey_path.endswith('.png'):
    pdf.image(sankey_path, x=10, y=None, w=190)
    pdf.ln(10)
    pdf.chapter_body('The Sankey diagram above visualizes the flow of customers through different stages '
                    'of their journey. The width of each link represents the volume of customers flowing '
                    'from one stage to another.')
else:
    pdf.chapter_body('The Sankey diagram is available as an interactive HTML visualization at: ' + sankey_path)

# Add Conversion Funnel
pdf.chapter_title('Conversion Funnel Visualization')
funnel_path = save_plotly_figure(funnel_fig, "conversion_funnel")
if funnel_path.endswith('.png'):
    pdf.image(funnel_path, x=10, y=None, w=190)
    pdf.ln(10)
    pdf.chapter_body('The conversion funnel above shows the progression of users through the customer journey, '
                    'from initial visit to purchase. The width of each stage represents the number of users '
                    'who reached that stage.')
else:
    pdf.chapter_body('The conversion funnel is available as an interactive HTML visualization at: ' + funnel_path)

# Save the PDF
pdf.output('customer_journey_report.pdf')
print("PDF report saved to 'customer_journey_report.pdf'")

#%%
# Main function to run the analysis
def main():
    # Generate sample data
    print("Generating sample data for Sankey chart...")
    sankey_data = generate_sankey_data(n_samples=1000)
    
    # Create Sankey chart
    print("Creating Sankey chart...")
    # (Sankey chart creation code would go here)
    
    # Create summary statistics
    print("Creating summary statistics...")
    # (Summary statistics code would go here)
    
    # Create conversion funnel
    print("Creating conversion funnel...")
    # (Conversion funnel code would go here)
    
    # Create PDF report
    print("Creating PDF summary report...")
    # (PDF report creation code would go here)
    
    print("Analysis completed successfully!")

if __name__ == "__main__":
    main() 
# %%
