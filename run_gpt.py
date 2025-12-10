#!/usr/bin/env python3
from openai import OpenAI

# Read your API key from file
with open("api.key", "r") as f:
    api_key = f.read().strip()

client = OpenAI(api_key=api_key)

system_prompt = "You are a helpful assistant."
prompt = "Hello! Give me a recipe for a chocolate cake."

response = client.chat.completions.create(
    model="gpt-5-nano",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
)

print(response.choices[0].message.content)
