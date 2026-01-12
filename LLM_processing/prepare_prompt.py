
import os
import sys
schema_dir = "schemas/"
schema_de_dir = schema_dir + "lang-de/"
import json
from pathlib import Path

def extract_section(text: str, header: str) -> list[str]:
    lines = text.splitlines()
    collecting = False
    result = []

    for line in lines:
        if line.strip() == header:
            collecting = True
            continue

        if collecting:
            if line.strip() == "":
                break
            result.append(line)

    return result

grammar_schema_text = Path("testing_validation/ref_schemas/grammar_schema.json").read_text(encoding="utf-8")
unimorph_schema_text = Path("testing_validation/ref_schemas/unimorph_schema.json").read_text(encoding="utf-8")     
custom_schema_text = Path("testing_validation/ref_schemas/custom_schema.json").read_text(encoding="utf-8")
annotation_file_text = Path("prompt_annotations/feature_annotations.txt").read_text(encoding="utf-8")
with open("testing_validation/ref_schemas/grammar_schema_verbose.txt", encoding="utf8") as f:
    grammar_schema_verbose = f.read()

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

            feats_annotation = extract_section(annotation_file_text, filename.split(".")[0])
            feats_annotation = "\n".join(feats_annotation)
            preferred_feats = Path(f"prompt_annotations/preferred_features/{filename.split('.')[0]}_preferred_features.txt").read_text(encoding="utf-8")
            preferred_feats = "\n".join(preferred_feats.splitlines()[1:])
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
                f"{grammar_schema_text}\n\n\n"

                f"{grammar_schema_verbose}\n"

                "SCHEMA GUIDELINES:\n"
                "- Use patterns for pattern-like grammar rules, endings if the paradigm consists of variations in endings, and forms if the paradigm is simply a specific set of forms."
                "- Assign titles to individual patterns if and only if there are multiple patterns under the same rule.\n"
                "- Assign titles to individual rules if and only if the following is the case:\n"
                "   - the rule in question is describing a tense\n"
                "   - the tense encodes more than one aspect\n"
                "   In that case, the title of the rule is the name by which the tense is commonly known in English.\n\n"

                "INSTRUCTIONS:\n"
                "- Use ONLY the information present in the provided data.\n"
                "- Rephrase and reformat content where necessary, but do NOT add new facts.\n"
                "- Merge overlapping information into a single coherent description in English.\n"
                "- Preserve the hierarchy and ordering of the reference schema.\n"
                "- Ignore any information that is not relevant to the topic at hand.\n\n"

                "FEATURES AND THEIR VALUES:\n"
                f"- Feature names and values are prefixed with either u- or x-.\n"
                "- \"u-\" is for UniMorph features and values, \"x-\" for custom (non-UniMorph, cross-linguistic) features and values.\n"
                "- Whenever possible, use UniMorph prefixes.\n"
                "- Invent new features only if no appropriate UniMorph features exists.\n"
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

                f"If possible, select features among the ones below:\n{preferred_feats}\n\n"
                f"The following may help guide you in your feature selection:\n{feats_annotation}\n\n\n"

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

            