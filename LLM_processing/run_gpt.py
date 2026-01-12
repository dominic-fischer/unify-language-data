#!/usr/bin/env python3
import sys
from openai import OpenAI
import subprocess
import os
from pathlib import Path
import subprocess
import sys


# Read your API key from file
with open("api.key", "r") as f:
    api_key = f.read().strip()

used_model = None

client = OpenAI(api_key=api_key)
langs = ["chewa", "shona", "swahili", "zulu", "french", "italian", "portuguese", "romanian", "spanish"]
schema_dir = "../schemas/"
schema_de_dir = schema_dir + "lang-de/"
save_dir = f"../outputs/outputs_{used_model}"
# create the save directory if it does not exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for lang in langs:
    for current_file in os.listdir(schema_de_dir):
        save_filename = f"{lang}_{current_file}_output.txt"
        # skip if file already exists
        if os.path.exists(f"{save_dir}/{save_filename}"):
            print(f"File '{save_dir}/{save_filename}' already exists, skipping...")
            continue
        
        prompt_dir = "../prompts/"
        file = prompt_dir + f"/{lang}_{current_file}_prompt.txt"
        # if the prompt file does not exist, skip
        if not os.path.exists(file):
            print(f"Prompt file '{file}' does not exist, skipping...")
            continue
     
        print(f"Processing '{lang}' with schema '{current_file}'...")

        with open(file, "r", encoding="utf-8") as f:
            content = f.read()

        # The system prompt is after SYSTEM PROMPT:, and the user prompt is after USER PROMPT:
        sections = content.split("USER PROMPT:")
        system_prompt = sections[0].replace("SYSTEM PROMPT:", "").strip()
        prompt = sections[1].strip()

        response = client.chat.completions.create(
            model=used_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )

        response = response.choices[0].message.content

        # save the response to a file
        with open(f"{save_dir}/{save_filename}", "w", encoding="utf-8") as f:
            f.write(response)

        # run the files testing_validation/_validate_grammar.py, _validate_grammar_extended.py, _validate_features.py to validate the output
        print(f"\tOutput saved to {save_dir}/{save_filename}")
        output_filepath = os.path.join(save_dir, save_filename)