import pandas as pd
import os

# Define paths to the folders
zero_shot_folder = 'Zero Shot results'
few_shot_folder = 'Few Shot results'
merged_folder = 'Merged results'

# Create the Merged results folder if it doesn't exist
os.makedirs(merged_folder, exist_ok=True)

# Iterate through the files in the zero-shot folder
for file_name in os.listdir(zero_shot_folder):
    if file_name.endswith('.csv'):
        # Construct paths for zero-shot and few-shot files
        zero_shot_path = os.path.join(zero_shot_folder, file_name)
        few_shot_path = os.path.join(few_shot_folder, file_name.replace('.csv', '_fewshot.csv'))

        # Read the zero-shot and few-shot CSV files
        zero_shot_df = pd.read_csv(zero_shot_path)
        few_shot_df = pd.read_csv(few_shot_path)

        # Merge the two DataFrames on the 'Query' column
        merged_df = pd.merge(zero_shot_df, few_shot_df, on='Query', suffixes=('_zero_shot', '_few_shot'))

        # Rename columns for clarity
        merged_df = merged_df.rename(columns={
            'Output_zero_shot': 'Zero Shot Output',
            'Output_few_shot': 'Few Shot Output'
        })

        # Save the merged DataFrame to the Merged results folder
        merged_file_path = os.path.join(merged_folder, file_name.replace('.csv', '_merged.csv'))
        merged_df.to_csv(merged_file_path, index=False)

        # Print confirmation
        print(f"Merged file saved: {merged_file_path}")

# Print the first few rows of the last merged DataFrame for verification
print(merged_df.head())