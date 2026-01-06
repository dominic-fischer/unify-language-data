
import os
import sys
schema_dir = "schemas/"
schema_de_dir = schema_dir + "lang-de/"
import json
from pathlib import Path

grammar_schema_text = Path("testing_validation/ref_schemas/grammar_schema.json").read_text(encoding="utf-8")

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

            prompt = (
                f"You are a linguist tasked with normalizing grammatical descriptions for the language {lang}.\n\n"

                f"The data you receive comes from multiple sources and relates to the following topics:\n"
                f"{', '.join(topic_terms)}.\n\n"

                "Your goal is to transform the data into a structure that EXACTLY matches the reference schema provided.\n\n"

                "REFERENCE SCHEMA (structure and headings must be followed exactly):\n"
                f"{grammar_schema_text}\n\n"

                "INSTRUCTIONS:\n"
                "- Use ONLY the information present in the provided data.\n"
                "- Rephrase content where necessary, but do NOT add new facts.\n"
                "- Merge overlapping information into a single coherent description.\n"
                f"- Make sure to ignore information that is not relevant to the topic(s) in question: {', '.join(topic_terms)}.\n"
                "- Preserve the hierarchy and ordering of the reference schema.\n\n"

                "FEATURES AND THEIR VALUES:\n"
                f"- Feature names and values are prefixed with either u-, x- or {langs_w_abbrevs[lang]}-.\n"
                "- Use the prefix \"u-\" for UniMorph features and values.\n"
                "- Use the prefix \"x-\" for custom (non-UniMorph, cross-linguistic) features and values.\n"
                f"- Use the language prefix \"{lang}-\" ONLY for language-specific features and values.\n"
                "- When multiple representations are possible, prefer:\n"
                "  1) UniMorph (\"u-\") features,\n"
                "  2) then custom (\"x-\") features,\n"
                f"  3) and only then language-specific (\"{lang}-\") features.\n"
                "- Do NOT invent new features if an appropriate UniMorph feature exists.\n\n"

                "SOURCE DATA:\n"
                f"{match_prompt}\n\n"

                "EXAMPLE OUTPUT (for German):\n"
                f"{reference_content}\n\n"

                "OUTPUT RULES:\n"
                "- Output ONLY the synthesized schema.\n"
                "- Do NOT include explanations, comments, or metadata.\n"
                "- Use the same formatting style as the example.\n"
                "- Do not take content from the example; it is only to illustrate the format.\n\n"
            )

            # save sys_prompt and prompt to a file for later use
            out_dir = "prompts/"
            os.makedirs(out_dir, exist_ok=True)
            out_filepath = os.path.join(out_dir, f"{lang}_{filename}_prompt.txt")
            with open(out_filepath, "w", encoding="utf-8") as out_f:
                out_f.write(f"SYSTEM PROMPT:\n{sys_prompt}\n\n")
                out_f.write(f"USER PROMPT:\n{prompt}\n")


            