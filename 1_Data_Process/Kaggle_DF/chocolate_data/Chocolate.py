#%%
import pandas as pd
import os

# Define the path to the CSV file
# Assuming the file is in the same directory as this script
file_path = os.path.join(os.path.dirname(__file__), "chocolate sales.csv")

try:
    # Read the CSV file
    chocolate_data = pd.read_csv(file_path)
    print(f"Successfully loaded chocolate dataset with {chocolate_data.shape[0]} rows and {chocolate_data.shape[1]} columns")
    
    # Display the first few rows of the dataset
    print("\nFirst few rows of the chocolate dataset:")
    print(chocolate_data.head())
    
    # Display basic information about the dataset
    print("\nDataset information:")
    print(chocolate_data.info())
    
except FileNotFoundError:
    print(f"Error: The file 'chocolate sales.csv' was not found at {file_path}")
    print("Please make sure the file exists in the same directory as this script.")
except Exception as e:
    print(f"Error reading the CSV file: {e}")

# %% 