from langchain_ollama import OllamaLLM
import pandas as pd

models = ['llama2']

sys_query = ""

for cur_model in models:
    model = OllamaLLM(model=cur_model)
    df = pd.read_csv("input.csv")
    output_df = pd.DataFrame(columns=['Query', 'Output'])

    for index, row in df.iterrows():
        query = row['Query']
        query = sys_query + query 
        output = model.invoke(query)
        print(output)
        
        new_row = pd.DataFrame({'Query': [query], 'Output': [output]})
        output_df = pd.concat([output_df, new_row], ignore_index=True)

    output_df.to_csv(f"{cur_model}.csv", index=False)