
import os
import sys
schema_dir = "schemas/"
schema_de_dir = schema_dir + "lang-de/"
import json
from pathlib import Path

grammar_schema_text = Path("testing_validation/ref_schemas/grammar_schema.json").read_text(encoding="utf-8")
unimorph_schema_text = Path("testing_validation/ref_schemas/unimorph_schema.json").read_text(encoding="utf-8")     
custom_schema_text = Path("testing_validation/ref_schemas/custom_schema.json").read_text(encoding="utf-8")

langs = ["chewa", "shona", "swahili", "zulu", "french", "italian", "portuguese", "romanian", "spanish"]
langs_w_abbrevs = {
    "chewa": "ny",
    "shona": "sn",
    "swahili": "sw",
    "zulu": "zu",
    "french": "fr",
    "italian": "it",
    "portuguese": "pt",
    "romanian": "ro",
    "spanish": "es"
}
mappings_docx_dir = "mappings_docx/"
mappings_wikipedia_dir = "mappings_wikipedia/"
langs_w_files = {lang: None for lang in langs}

for lang in langs_w_files:
    langs_w_files[lang] = []
    for filename in os.listdir(mappings_docx_dir):
        if lang in filename.lower():
            langs_w_files[lang].append(mappings_docx_dir + filename)
    for filename in os.listdir(mappings_wikipedia_dir):
        if lang in filename.lower():
            langs_w_files[lang].append(mappings_wikipedia_dir + filename)

# go through all schema_de files
for lang, mapping_file_paths in langs_w_files.items():

    sys_prompt = f"You are an expert in the {lang} language. You excel at inspecting linguistic data from different sources and distilling them into a specific format."

    for filename in os.listdir(schema_de_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(schema_de_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                reference_content = f.read()
            docx_matches = []
            wikipedia_matches = []
            for mapping_file_path in mapping_file_paths:
                # open the file and load as json
                with open(mapping_file_path, "r", encoding="utf-8") as f:
                    mapping_data = json.load(f)
                    relevant_entry = mapping_data["schemas\\lang-de\\" + filename]
                    matches = relevant_entry.get("matches", [])
                    topic_terms = relevant_entry.get("topic_terms", [])
                    strategy = relevant_entry.get("strategy", "unknown")
                    for match in matches:
                        # skip vocabulary matches
                        if match["full"].startswith("## Vokabular >"):
                            continue
                        # skip improbable matches
                        if strategy == "embedding" and match["similarity"] < 0.5:
                            continue
                        if "wikipedia" in mapping_file_path:
                            wikipedia_matches.append(match["content"])
                        else:
                            docx_matches.append(match["content"])
            
            match_prompt = ""
            source_ind = 1
            if docx_matches:
                docx_joined = '\n\n'.join(docx_matches).lstrip("\n")
                match_prompt = f"Source {source_ind}:\n{docx_joined}\n\n" + "- "*50 + "\n\n"
                source_ind += 1
            if wikipedia_matches:
                wiki_joined = '\n\n'.join(wikipedia_matches).lstrip("\n")
                match_prompt += f"Source {source_ind}:\n{wiki_joined}\n\n" + "- "*50 + "\n\n"
            if match_prompt == "":
                # write an empty file to outputs
                continue

            # 1) Stable prefix: keep this EXACTLY identical across requests for caching benefits
            prompt_prefix = (
                "You are a linguist that specialises in distilling linguistic information. Your goal is to transform the data into a structure that satisfies the constraints specified in the validation schema provided.\n\n"

                "VALIDATION SCHEMA (structure and headings must be followed exactly):\n"
                f"{grammar_schema_text}\n\n"

                "The schema consists of RULES, each applying to a specific grammatical FEATURE and its ATTRIBUTE.\n"

                "Each Rule must contain:\n"
                "- A name: built by joining feature-attribute pairs with an underscore, and a double underscore between different pairs.\n"
                "- the 'applies' field: specifies to which feature-attribute pair the rule applies. Note that no two rules should have the same 'applies' field. In that case, the two rules should be combined to form one with a 'patterns' field.\n"
                "- either a field 'pattern' or 'patterns' or 'forms':\n"
                "   - a 'pattern' is a string-like template\n"
                "   - 'patterns' is a list of sub-patterns, that each contain either 'pattern' or 'forms', and may include 'examples', 'notes' or 'endings.\n"
                "   - 'forms' is a list of specific word forms. It must contain 'features' and either 'form' or 'pattern'. It may also include a 'note' field.\n"
                "- it may include the fields 'notes', 'examples' or 'endings'.\n"
                "   - 'endings' must include 'features' and 'form', and may include a 'note' field.\n"

                "The 'Features' field on the same level as 'Rules' is the set of all features and their attributes used in the rules.\n"
                "There are also other optional fields at the top level, 'Negation' and 'Usage'. 'Usage' is used to specify usage of a tense, 'Negation' is used to specify negation in that tense. They should only be used in schemas pertaining to tenses. Both may contain 'pattern', 'note' or 'examples' fields, 'Usage' may additionally contain 'applies'.\n\n"

                "INSTRUCTIONS:\n"
                "- Use ONLY the information present in the provided data.\n"
                "- Rephrase and reformat content where necessary, but do NOT add new facts.\n"
                "- Merge overlapping information into a single coherent description in English.\n"
                "- Preserve the hierarchy and ordering of the reference schema.\n"
                "- Ignore any information that is not relevant to the topic at hand.\n\n"

                "FEATURES AND THEIR VALUES:\n"
                f"- Feature names and values are prefixed with either u-, x- or the language's abbreviation as a prefix.\n"
                f"- Use the language prefix ONLY for language-specific features and values.\n"
                "- Use the prefix \"u-\" for UniMorph features and values.\n"
                "- Use the prefix \"x-\" for custom (non-UniMorph, cross-linguistic) features and values.\n"
                "- When multiple representations are possible, prefer:\n"
                "  1) UniMorph (\"u-\") features,\n"
                "  2) then custom (\"x-\") features,\n"
                f"  3) and only then language-specific features.\n\n"
                "- Do NOT invent new features if an appropriate UniMorph feature exists.\n\n"

                "Below are the features for reference:\n"
                "UNIMORPH FEATURES AND VALUES:\n"
                f"{unimorph_schema_text}\n\n"
                "CUSTOM FEATURES AND VALUES:\n"
                f"{custom_schema_text}\n\n"

                "OUTPUT FORMAT CONSTRAINTS:\n"
                "- Output ONLY the synthesized schema.\n"
                "- The output must be in a human-readable, YAML-like text format (as in the example).\n"
                "- The output is NOT JSON.\n"
                "- The structure, keys, nesting, and values must correspond exactly to the provided validation schema.\n"
                "- The output must be directly convertible to valid JSON without adding, removing, or reinterpreting information.\n"
                "- Do NOT include empty objects or sections unless they are required by the schema.\n"
                "- Do NOT include explanations, comments, or metadata.\n"
                "- Do not take content from the example; it only illustrates the format.\n\n\n"
            )

            # 2) Per-file suffix: this is the part that changes per input file / run
            prompt_suffix = (
                f"You are a now tasked with normalizing grammatical descriptions for the language {lang} according to the guidelines outlined above.\n\n"

                f"The data you receive comes from multiple sources and relates to the following topics:\n"
                f"{', '.join(topic_terms)}.\n\n"

                "SOURCE DATA:\n"
                f"{match_prompt}\n\n"

                "EXAMPLE OUTPUT (for German):\n"
                "Use this ONLY as an illustration of the format, do NOT copy any content from it.\n"
                f"{reference_content}\n\n"
            )

            prompt = prompt_prefix + prompt_suffix



            # save sys_prompt and prompt to a file for later use
            out_dir = "prompts/"
            os.makedirs(out_dir, exist_ok=True)
            out_filepath = os.path.join(out_dir, f"{lang}_{filename}_prompt.txt")
            with open(out_filepath, "w", encoding="utf-8") as out_f:
                out_f.write(f"SYSTEM PROMPT:\n{sys_prompt}\n\n")
                out_f.write(f"USER PROMPT:\n{prompt}\n")


            