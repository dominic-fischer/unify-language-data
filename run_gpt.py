#!/usr/bin/env python3
import sys
from openai import OpenAI

# Read your API key from file
with open("api.key", "r") as f:
    api_key = f.read().strip()

client = OpenAI(api_key=api_key)

lang = "chewa"
save_dir = f"outputs"
current_file = "FORMAT_adjective.txt"
save_filename = f"{lang}_{current_file}_output.txt"

prompt_dir = "prompts/"
file = prompt_dir + f"/{lang}_{current_file}_prompt.txt"
with open(file, "r", encoding="utf-8") as f:
    content = f.read()

# The system prompt is after SYSTEM PROMPT:, and the user prompt is after USER PROMPT:
sections = content.split("USER PROMPT:")
system_prompt = sections[0].replace("SYSTEM PROMPT:", "").strip()
prompt = sections[1].strip()

response = client.chat.completions.create(
    model="gpt-5-nano",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
)

response = response.choices[0].message.content

# save the response to a file
with open(f"{save_dir}/{save_filename}", "w", encoding="utf-8") as f:
    f.write(response)
