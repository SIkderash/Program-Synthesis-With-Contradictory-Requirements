import os
import pandas as pd

folder_path = r"C:/Users/Shanto/Desktop/Merged results"
updated_folder_path = r"C:/Users/Shanto/Desktop/Final results"

# Ensure output folder exists
os.makedirs(updated_folder_path, exist_ok=True)

# Function to categorize the model's response
def categorize_output(output):
    if isinstance(output, str):  # Check if the value is a string
        if "contradiction" in output.lower():
            if "def " in output or "function" in output:  # Checks if a function is also provided
                return 2
            return 1
    return 0

# Process each CSV file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):  # Process only CSV files
        file_path = os.path.join(folder_path, filename)
        
        # Read CSV file
        df = pd.read_csv(file_path)

        # Apply categorization function to both columns
        df["Zero_Shot_Contradiction"] = df["Zero Shot Output"].apply(categorize_output)
        df["Few_Shot_Contradiction"] = df["Few Shot Output"].apply(categorize_output)

        # Save updated file
        updated_file_path = os.path.join(updated_folder_path, filename)
        df.to_csv(updated_file_path, index=False)

print("Processing completed! Updated files are saved in:", updated_folder_path)
