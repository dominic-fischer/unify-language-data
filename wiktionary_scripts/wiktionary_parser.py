import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

def print_data(data, indent=0):
    prefix = "  " * indent
    if isinstance(data, dict):
        for k, v in data.items():
            print(f"{prefix}{k}:")
            print_data(v, indent + 1)
    elif isinstance(data, list):
        for item in data:
            print_data(item, indent + 1)
    else:
        print(f"{prefix}{data}")

def extract_words(jsonl_file, max_words=10000, lang=None):
    words = []
    counter = 0
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            if lang is not None:
                if "lang" not in entry or entry['lang'] != lang:
                    continue
            words.append(entry)
            counter += 1
            if counter >= max_words:
                break
    return words

# Set language and max word count
lang = "Portuguese"
max_words = 10
words = extract_words("wiktionary_data/raw-wiktextract-data.jsonl", max_words=max_words, lang=lang)

# Print each word systematically
for word in words:
    for k, v in word.items():
        if type(v) == list:
            print(k)
            if k == "senses":
                for sens_dict in v:
                    items = list(sens_dict.items())
                    for idx, (key, value) in enumerate(items):
                        is_first = (idx == 0)
                        is_last = (idx == len(items) - 1)

                        prefix = "\t{" if is_first else "\t\t"
                        suffix = "}" if is_last else ""
                        print(f"{prefix}{key}: {value}{suffix}")

            else:
                for item in v:
                    print("\t", item)
        else:
            print(k, ":", v)
    #print_data(word)
    print("-" * 50)

print(f"Total words in {lang}: {len(words)}, capped at {max_words}")
