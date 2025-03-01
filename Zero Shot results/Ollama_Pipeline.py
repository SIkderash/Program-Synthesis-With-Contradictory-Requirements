import json
import pandas as pd
from langchain_ollama import OllamaLLM

models = ['qwen2.5-coder:3b', 'qwen2.5-coder:latest', 'qwen2.5-coder:14b', 'qwen2.5:3b', 'qwen2.5:7b', 'qwen2.5:14b', 'codellama:13b']

isZeroShot = True
run_status_file_name = "run_status.json"

if isZeroShot == False:
    run_status_file_name = "run_status_fewshot.json"


def addSysPrompt(query):
    return f"""You are an assistant to help synthesize programs from a given requirement. You will be given a requirement below. Try to synthesize a program for it. 

            Requirement:
            {query}
            
            (
            The requirement may or may not be correct, by logical, mathematical or other means. You need to figure out whether it is a contradictory requirement or not. 
            If it is contradictory, then respond by 'The requirement has contradictions, thus I'm unable to provide a program' and then describe what the contradiction is.'
            Otherwise, if there is no contradiction, just simply provide a program. Below are are some examples for you:
            
            Requirement:
            "Write a function that checks if all numbers in a list are even and returns True if all numbers are even, otherwise returns False."
            Response:
            def check_all_even(numbers):
                return all(number % 2 == 0 for number in numbers)
            
            Requirement:
            "Write a function that returns True if a light is both RED and BLUE at the same time."
            Response:
            Contradiction: A light cannot be both RED and BLUE simultaneously. These are mutually exclusive states.
            Type: Contradictory (mutually exclusive opposites).
            Subtype: Simplex (no conditions, direct contradiction).

            Requirement:
            "Write a function that returns True if the heater is both ON and OFF during the same season."
            Response:
            Contradiction: The heater cannot be both ON and OFF at the same time, even under the same condition (season).
            Type: Contradictory (mutually exclusive opposites).
            Subtype: Idem (same condition applied to both states).

            Requirement:
            "Write a function that returns True if the traffic light is RED during rush hour and GREEN during off-peak times."
            Response:
            Contradiction: The traffic light cannot be both RED and GREEN at the same time, even under different conditions (rush hour and off-peak times).
            Type: Contradictory (mutually exclusive opposites).
            Subtype: Alius (different conditions applied to both states).

            Requirement:
            "Write a function that returns True if a car is both MOVING and PARKED at the same time."
            Response:
            Contradiction: A car cannot be both MOVING and PARKED simultaneously. These are contrary states.
            Type: Contrary (mutually inconsistent but not exhaustive).
            Subtype: Simplex (no conditions, direct contradiction).

            Requirement:
            "Write a function that returns True if the security system is both ARMED during the night and DISARMED during the day."
            Response:
            Contradiction: The security system cannot be both ARMED and DISARMED at the same time, even under the same condition (time of day).
            Type: Contrary (mutually inconsistent but not exhaustive).
            Subtype: Idem (same condition applied to both states).

            Requirement:
            "Write a function that returns True if the AC is both COOLING during the day and HEATING during the night."
            Response:
            Contradiction: The AC cannot be both COOLING and HEATING at the same time, even under different conditions (day and night).
            Type: Contrary (mutually inconsistent but not exhaustive).
            Subtype: Alius (different conditions applied to both states).

            Requirement:
            "Write a function that checks if all students in a list have passed and returns True if some students have passed."
            Response:
            Contradiction: The universal statement ("all students have passed") is contradicted by the conditional statement ("some students have passed").
            Type: Subalterns (dependency between universal and conditional statements).
            Subtype: Simplex (no conditions, direct contradiction).

            Requirement:
            "Write a function that checks if all numbers are even and returns True if some numbers are even."
            Response:
            Contradiction: The universal statement ("all numbers are even") is contradicted by the conditional statement ("some numbers are even").
            Type: Subalterns (dependency between universal and conditional statements).
            Subtype: Idem (same condition applied to both statements).

            Requirement:
            "Write a function that checks if all cars are electric and returns True if some cars are electric but only if they are new."
            Response:
            Contradiction: The universal statement ("all cars are electric") is contradicted by the conditional statement ("some cars are electric but only if they are new").
            Type: Subalterns (dependency between universal and conditional statements).
            Subtype: Alius (different conditions applied to both statements).

            Requirement:
            "Write a function that checks if some numbers in a list are even and returns True if at least one number is even, otherwise returns False."
            Response:
            def check_some_even(numbers):
                return any(number % 2 == 0 for number in numbers)
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
    df = pd.read_csv("input1.csv")
    output_df = pd.DataFrame(columns=['Query', 'Output'])
    
    for _, row in df.iterrows():
        query = row['Query']
        final_query = query
        if isZeroShot == False:
            final_query = addSysPrompt(query) 
        output = model.invoke(final_query)
        print(output)
        
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