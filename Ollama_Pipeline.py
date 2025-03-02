import json
import pandas as pd
from langchain_ollama import OllamaLLM

models = ['qwen2.5-coder:3b', 'qwen2.5-coder:latest', 'qwen2.5-coder:14b', 'qwen2.5:3b', 'qwen2.5:7b', 'qwen2.5:14b', 'codellama:13b', 
          'llama2:latest', 'gemma2', 'codegemma'] #'deepseek-r1:7b', 'deepseek-coder:6.7b',

isZeroShot = False
run_status_file_name = "run_status.json"

if isZeroShot == False:
    run_status_file_name = "run_status_fewshot.json"


def addSysPrompt(query):
    return f"""You are an assistant to help synthesize programs from a given requirement. You will be given a requirement below. Try to synthesize a program for it. 

            Requirement:
            {query}
            
            (
            The requirement may or may not be correct, by logical, mathematical or other means. You need to figure out whether it is a contradictory requirement or not. 
            If it is contradictory, then first respond by 'Contradiction Found' and then describe what the contradiction is.'
            Otherwise, if there is no contradiction, just simply provide a program. Below are are some examples for you:
            
            Requirement:
            "Write a function that checks if all numbers in a list are even and returns True if all numbers are even, otherwise returns False."
            Response:
            def check_all_even(numbers):
                return all(number % 2 == 0 for number in numbers)
            
            Requirement:
            "Write a function that returns True if a light is both RED and BLUE at the same time."
            Response:
            Contradiction Found: A light cannot be both RED and BLUE simultaneously. These are mutually exclusive states.

            Requirement:
            "Write a function that returns True if the heater is both ON and OFF during the same time."
            Response:
            Contradiction: The heater cannot be both ON and OFF at the same time, even under the same condition (season).

            Requirement:
            "Write a function that returns True if the traffic light is RED during rush hour and GREEN during off-peak times."
            Response:
            def is_correct_traffic_light(color, time_of_day):
                if time_of_day == "rush_hour":
                    return color == "RED"
                elif time_of_day == "off_peak":
                    return color == "GREEN"
                return False  # Default case if an unknown condition is provided
            )"""


try:
    with open(run_status_file_name, "r") as file:
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
    df = pd.read_csv("input.csv")
    output_df = pd.DataFrame(columns=['Query', 'Output'])
    
    for _, row in df.iterrows():
        query = row['Query']
        final_query = query
        if isZeroShot == False:
            final_query = addSysPrompt(query) 
        output = model.invoke(final_query)
        # print(output)
        
        new_row = pd.DataFrame({'Query': [query], 'Output': [output]})
        output_df = pd.concat([output_df, new_row], ignore_index=True)

    output_file = f"{index}.csv"
    if isZeroShot == False:
        output_file = f"{index}_fewshot.csv"

    output_df.to_csv(output_file, index=False)

    run_status[cur_model] = 1  
    with open(run_status_file_name, "w") as file:
        json.dump(run_status, file, indent=4)

    index += 1