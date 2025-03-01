import os
import pandas as pd

# Define the folder containing the CSV files
results_folder = "Final results"

# Initialize a list to store the results
results = []

# Iterate through each file in the folder
for filename in os.listdir(results_folder):
    if filename.endswith(".csv"):
        file_path = os.path.join(results_folder, filename)
        df = pd.read_csv(file_path)

        # Calculate metrics for Zero Shot
        zero_shot_counts = df["Zero_Shot_Contradiction"].value_counts().reindex([0, 1, 2], fill_value=0)
        zero_shot_accuracy = zero_shot_counts[1] / len(df)

        # Calculate metrics for Few Shot
        few_shot_counts = df["Few_Shot_Contradiction"].value_counts().reindex([0, 1, 2], fill_value=0)
        few_shot_accuracy = few_shot_counts[1] / len(df)

        # Append the results for this file
        results.append({
            "Model": filename.replace(".csv", ""),
            "Zero_Shot_0s": zero_shot_counts[0],
            "Zero_Shot_1s": zero_shot_counts[1],
            "Zero_Shot_2s": zero_shot_counts[2],
            "Zero_Shot_Accuracy": zero_shot_accuracy,
            "Few_Shot_0s": few_shot_counts[0],
            "Few_Shot_1s": few_shot_counts[1],
            "Few_Shot_2s": few_shot_counts[2],
            "Few_Shot_Accuracy": few_shot_accuracy
        })

# Convert the results to a DataFrame
results_df = pd.DataFrame(results)

# Save the consolidated results to a new CSV file
output_file = "consolidated_results.csv"
results_df.to_csv(output_file, index=False)

print(f"Consolidated results saved to {output_file}")