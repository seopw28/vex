#%% Imports
import pyupbit
import pandas as pd
import pprint
import matplotlib.pyplot as plt
pd.options.display.float_format = "{:.0f}".format

#%% Data Collection
# Get 30 days of daily data
df = pyupbit.get_ohlcv(ticker="KRW-BTC", 
                       interval="day",  # daily data
                       count=30  # last 30 days
                       )

# Name the index (date column)
df.index.name = "Date"
df.head()

#%% Data Visualization
# Create line graph with improved styling
plt.figure(figsize=(15, 8))
plt.plot(df.index, df['close'], color='skyblue', linewidth=2, marker='o')
plt.title('Bitcoin Daily Closing Prices (Last 30 Days)', fontsize=16, pad=15)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Price (KRW)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)

# Format y-axis with comma separator for better readability
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

plt.tight_layout()
plt.show()

#%% Data Export
df.to_csv("test_btc.csv")
print(df)

#%% PDF Report Generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# Create a PDF document
doc = SimpleDocTemplate("Bitcoin_Analysis.pdf", pagesize=letter)
styles = getSampleStyleSheet()
story = []

# Add title
title = Paragraph("Bitcoin Analysis Report", styles['Title'])
story.append(title)
story.append(Spacer(1, 12))

# Add summary of the analysis
summary = Paragraph("This report contains the analysis of Bitcoin prices over the last 30 days.", styles['Normal'])
story.append(summary)
story.append(Spacer(1, 12))

# Add the DataFrame as a table
from reportlab.platypus import Table, TableStyle
data = [df.columns.tolist()] + df.values.tolist()
table = Table(data)
table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), 'grey'),
                           ('TEXTCOLOR', (0, 0), (-1, 0), 'white'),
                           ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                           ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                           ('FONTSIZE', (0, 0), (-1, 0), 14),
                           ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                           ('BACKGROUND', (0, 1), (-1, -1), 'white'),
                           ('TEXTCOLOR', (0, 1), (-1, -1), 'black'),
                           ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                           ('FONTSIZE', (0, 1), (-1, -1), 12),
                           ('GRID', (0, 0), (-1, -1), 1, 'black'),
                           ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                           ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),]))
story.append(table)

# Build the PDF
doc.build(story)
print("PDF report generated: Bitcoin_Analysis.pdf")

# %%
