import pandas as pd
df = pd.read_csv('input.csv')
dropped_df = df.drop('Example', axis='columns')
dropped_df.to_csv("input1.csv")