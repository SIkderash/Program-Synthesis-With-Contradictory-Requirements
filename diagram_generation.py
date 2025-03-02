import os
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# Create the "Result diagrams" folder if it doesn't exist
os.makedirs("Result diagrams", exist_ok=True)

df = pd.read_csv("consolidated_results.csv")

# Plot for Zero Shot
plt.figure(figsize=(12, 6))
plt.bar(df["Model"], df["Zero_Shot_0s"], color='#FF5733', label="0s (No Contradiction Detected)")
plt.bar(df["Model"], df["Zero_Shot_1s"], bottom=df["Zero_Shot_0s"], color='#33FF57', label="1s (Contradiction Detected)")
plt.bar(df["Model"], df["Zero_Shot_2s"], bottom=df["Zero_Shot_0s"] + df["Zero_Shot_1s"], color='#3357FF', label="2s (Contradiction Detected Yet Code Provided)")
plt.xlabel("Model")
plt.ylabel("Count")
plt.title("Zero Shot: Distribution of Contradiction Predictions")
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig("Result diagrams/zero_shot_distribution.png")  # Save the plot
plt.close()

# Plot for Few Shot
plt.figure(figsize=(12, 6))
plt.bar(df["Model"], df["Few_Shot_0s"], color='#FF5733', label="0s (No Contradiction Detected)")
plt.bar(df["Model"], df["Few_Shot_1s"], bottom=df["Few_Shot_0s"], color='#33FF57', label="1s (Contradiction Detected)")
plt.bar(df["Model"], df["Few_Shot_2s"], bottom=df["Few_Shot_0s"] + df["Few_Shot_1s"], color='#3357FF', label="2s (Contradiction Detected Yet Code Provided)")
plt.xlabel("Model")
plt.ylabel("Count")
plt.title("Few Shot: Distribution of Contradiction Predictions")
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig("Result diagrams/few_shot_distribution.png")  # Save the plot
plt.close()

# Scatter plot for Few Shot Accuracy
plt.figure(figsize=(10, 6))
plt.scatter(df["Model"], df["Few_Shot_Accuracy"], color='b', s=100)  # s is the size of the points
plt.xlabel("Model")
plt.ylabel("Few Shot Accuracy")
plt.title("Few Shot Accuracy Across Models")
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.tight_layout()
plt.savefig("Result diagrams/few_shot_accuracy_scatter.png")  # Save the plot
plt.close()

# Heatmap for Accuracy Comparison
# heatmap_data = df[["Model", "Zero_Shot_Accuracy", "Few_Shot_Accuracy"]].set_index("Model")
# plt.figure(figsize=(8, 6))
# sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".2f")
# plt.title("Accuracy Comparison: Zero Shot vs. Few Shot")
# plt.tight_layout()
# plt.savefig("Result diagrams/accuracy_comparison_heatmap.png")  # Save the plot
# plt.close()