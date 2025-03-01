import os
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# Create the "Result diagrams" folder if it doesn't exist
os.makedirs("Result diagrams", exist_ok=True)

# Data
data = {
    "Model": ["qwen2.5-coder:3b", "qwen2.5-coder:latest", "qwen2.5-coder:14b", "qwen2.5:3b", "qwen2.5:7b", "qwen2.5:14b", "codellama:13b", "llama2:latest"],
    "Zero_Shot_0s": [89, 92, 91, 87, 86, 81, 91, 87],
    "Zero_Shot_1s": [0, 0, 0, 0, 0, 0, 0, 0],
    "Zero_Shot_2s": [3, 0, 1, 5, 6, 11, 1, 5],
    "Few_Shot_0s": [2, 1, 0, 1, 0, 0, 49, 47],
    "Few_Shot_1s": [61, 65, 61, 25, 12, 75, 16, 13],
    "Few_Shot_2s": [29, 26, 31, 66, 80, 17, 27, 32],
    "Zero_Shot_Accuracy": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "Few_Shot_Accuracy": [0.663, 0.707, 0.663, 0.272, 0.130, 0.815, 0.174, 0.141]
}

df = pd.DataFrame(data)

# Plot for Zero Shot
plt.figure(figsize=(12, 6))
plt.bar(df["Model"], df["Zero_Shot_0s"], label="0s (No Contradiction)")
plt.bar(df["Model"], df["Zero_Shot_1s"], bottom=df["Zero_Shot_0s"], label="1s (Correct Contradiction)")
plt.bar(df["Model"], df["Zero_Shot_2s"], bottom=df["Zero_Shot_0s"] + df["Zero_Shot_1s"], label="2s (Incorrect Contradiction)")
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
plt.bar(df["Model"], df["Few_Shot_0s"], label="0s (No Contradiction)")
plt.bar(df["Model"], df["Few_Shot_1s"], bottom=df["Few_Shot_0s"], label="1s (Correct Contradiction)")
plt.bar(df["Model"], df["Few_Shot_2s"], bottom=df["Few_Shot_0s"] + df["Few_Shot_1s"], label="2s (Incorrect Contradiction)")
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
heatmap_data = df[["Model", "Zero_Shot_Accuracy", "Few_Shot_Accuracy"]].set_index("Model")
plt.figure(figsize=(8, 6))
sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Accuracy Comparison: Zero Shot vs. Few Shot")
plt.tight_layout()
plt.savefig("Result diagrams/accuracy_comparison_heatmap.png")  # Save the plot
plt.close()