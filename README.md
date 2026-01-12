# Unify Language Data – Streamlit App

The application in this repository can be accessed here:  
👉 **[Unify Language Data: Streamlit App](https://dominic-fischer-unify-language-data-app-qdypkn.streamlit.app/)**

---

## Application Guide

![The landing page](imgs/Interface.png)

The interface is split into two main sections: **Grammar** and **Vocab**.  
Each section offers two views: **Browsing** and **Compare**.

---

## Grammar

### Browsing

The grammar browsing view allows users to select a language, a grammar topic, and additional fine-grained criteria in order to find specific grammar rules. Selecting a rule expands it below and displays all available details.

![The grammar browsing view](imgs/Grammar_browsing.png)

### Compare

The grammar compare view is best suited for comparing grammatical phenomena across two or more languages. The example below shows a comparison of the comparative construction in German and Romanian.

![The grammar compare view](imgs/Grammar_compare.png)

---

## Vocab

### Browsing

The vocab browsing view lets users filter words by part of speech and, optionally, by topic or etymological origin. Clicking on a word expands it below to show further details.

![The vocab browsing view](imgs/Vocab_browsing.png)

### Compare

The vocab compare view works analogously to the grammar compare view. The main difference is that vocabulary comparison is currently limited to two languages.

![The vocab compare view](imgs/Vocab_compare.png)

---

## Limitations

The most obvious limitation of the application is **data quality**:

- **A)** The source data (Wikipedia, Wiktionary, textbooks) is not always fully reliable or consistent.
- **B)** GPT-5.2 may have misinterpreted or inconsistently annotated certain phenomena, despite careful prompting.

More generally, comparing grammar across different languages — and especially across different language families — is inherently difficult. While considerable effort was made to ensure cross-lingual compatibility, the LLM-based processing inevitably introduced noise and unpredictability. This can result in the same phenomenon being annotated differently across languages, or even inconsistently within a single language.

Finally, GPT was instructed to work strictly with the provided source material in order to avoid hallucinations. While this improves faithfulness to the sources, it also means that some grammar descriptions remain incomplete.

---

## Background

### What? – The Idea

The aim of this project is to synthesize existing linguistic description data into a structured, uniform format with the help of an LLM. The resulting interface allows users to browse and compare the two core components of language learning — **grammar** and **vocabulary** — across multiple languages.

Currently supported languages include:

- **Romance languages**: French, Italian, Spanish, Portuguese, Romanian  
- **Bantu languages**: Chichewa (Chewa, Nyanja), Chishona (Shona), Kiswahili (Swahili), isiZulu (Zulu)

---

### Why? – The Motivation

Learning becomes more efficient when learners can recognize patterns, connect new information to what they already know, and evaluate similarities and differences between concepts. These processes help structure new information and support generalization and prediction.

Applied to language learning, this means that understanding a new language benefits greatly from placing it in the context of languages one already knows. However, traditional learning materials usually focus on a single target language. In addition, linguistic descriptions vary widely between textbooks and articles, reflecting different perspectives and organizational choices.

This project was designed to address these issues by:

- **A)** presenting linguistic data in a uniform and cross-lingually compatible way  
- **B)** allowing new linguistic information to be contextualized within a wider ecosystem of languages  
- **C)** enabling flexible and dynamic exploration of the data through filtering and comparison  

---

### How? – The Process

#### Data Preparation

Grammar data was sourced from textbook-style Word documents as well as Wikipedia articles. To make this data suitable for LLM processing, it was first converted into a unified, markdown-like format used consistently across sources. An example (from *Wikipedia: Italian Grammar*) is shown below:

```
    # Italian grammar

    Italian grammar is the body of rules describing the properties of the  Italian language. Italian words can be divided into the following lexical categories: articles, nouns, adjectives, pronouns, verbs, adverbs, prepositions, conjunctions, and interjections.

    ## Articles

    Italian articles vary according to definiteness (definite, indefinite, and partitive), number, gender, and the initial sound of the subsequent word. Partitive articles compound the preposition di with the corresponding definite article, to express uncertain quantity. In the plural, they typically translate into English as 'few'; in the singular, typically as 'some'.
```

For vocabulary data, the **Kaikki Wiktionary Archive** was used directly. Wiktionary’s entry structure served as the gold standard for vocabulary representation, even though it is not entirely consistent. Future work could involve augmenting this data with additional vocabulary sources or linking vocabulary more closely to grammar (for example, connecting inflection patterns to the words that follow them).

The next step was defining a **gold-standard schema** for grammar descriptions that could be applied consistently across languages. Designing this schema required several iterations before arriving at a structure that was both simple and robust. The schema can be represented as follows:

```
    Title
        Features
            FeatureName1: [value1, value2, ...]
            FeatureName2: [value1, value2, ...]

        Rules
            RuleName
                title?
                notes?
                applies
                    feature1: value
                    feature2: value
                (patterns | formparadigms | endingparadigms)+

    # This is the substructure of the three main components
    patterns               formparadigms               endingparadigms
        - pattern               - formparadigm              - endingparadigm
                                    - features                  - features  
                                    form                        form
        title?                  title?                      title?
        notes?                  notes?                      notes?
        examples?               examples?                   examples?

```

Eighteen grammar categories were defined, and for each of them a detailed **German reference schema** was created to guide the LLM. These reference schemas can be found in `schemas/lang-de/`.

---

#### Data Generation

Before generation, relevant sections of the source material were aligned with the corresponding grammar schemas. This step was crucial to avoid overwhelming the model with irrelevant information.

Section matching was performed using:

- keyword matching in titles and headings (for example, *determiner*, *article* for the “Determiners” schema), and  
- sentence-embedding similarity (cosine similarity), with a cutoff value of 0.5 to discard weak matches.

The prompt provided to GPT-5.2 included:

- general instructions for handling the input data  
- the validation schema in both JSON and text-based form  
- guidelines for implementing features and values  
- output-format constraints  
- topic-specific instructions for each grammar category  
- the aligned source sections  
- a German gold-standard output as reference  

The generated, standardized grammar descriptions can be found in `outputs_gpt-5.2/`, with the corresponding prompts located in `prompts/`.

---

#### Data Postprocessing

All generated schemas were subsequently validated and, where necessary, corrected. The validation and fixing process can be found in `testing_validation/`.

---

#### Interface Building

The interface was built using **Streamlit**. Keeping the schema relatively simple was essential to ensure that the data could be read and displayed efficiently.

One major issue that is not yet fully resolved is application performance. The Wiktionary vocabulary JSONL files in particular are quite large and are included as GitHub release assets so that the Streamlit app can access them. Iterating over these files still takes a noticeable amount of time, and finding a faster solution remains an area for future improvement.

---

### Footnotes

1. Grammar descriptions also exist for German. Unlike the others, these were created manually and serve as a gold standard for how the grammar schemas should look.  
2. The textbook data was compiled by the author over several years from various sources, especially for lower-resource Bantu languages.  
3. Grammar categories include: Adjectives, Adverbs, Agreement, Clauses, Conditionals, Constituents, Derivation, Determiners, Future Tense, Imperative, Negation, Nouns, Passive Voice, Past Tense, Present Tense, Pronouns, Reported Speech, Subjunctive.