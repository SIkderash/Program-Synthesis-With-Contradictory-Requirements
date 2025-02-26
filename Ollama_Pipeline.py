import json
import pandas as pd
from langchain_ollama import OllamaLLM

models = ['qwen2.5-coder:3b', 'qwen2.5-coder:7b', 'qwen2.5-coder:14b', 'qwen2.5:3b', 'qwen2.5:7b', 'qwen2.5:14b', 'codellama:13b']
sys_query = ""

# Initialize or load the run_status.json file
try:
    with open("run_status.json", "r") as file:
        run_status = json.load(file)
except FileNotFoundError:
    run_status = {}

index = 0
for cur_model in models:
    if cur_model in run_status and run_status[cur_model] == 1:
        print(f"Skipping {cur_model} as it has already been processed.")
        index += 1  
        continue

    model = OllamaLLM(model=cur_model)
    df = pd.read_csv("input1.csv")
    output_df = pd.DataFrame(columns=['Query', 'Output'])

    for _, row in df.iterrows():
        query = row['Query']
        query = sys_query + query 
        output = model.invoke(query)
        print(output)
        
        new_row = pd.DataFrame({'Query': [query], 'Output': [output]})
        output_df = pd.concat([output_df, new_row], ignore_index=True)

    output_df.to_csv(f"{index}.csv", index=False)

    run_status[cur_model] = 1  
    with open("run_status.json", "w") as file:
        json.dump(run_status, file, indent=4)

    index += 1