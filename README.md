This repository's application can be found under: [Unify Language Data: Streamlit App](https://dominic-fischer-unify-language-data-app-qdypkn.streamlit.app/)

## Application Guide
![The landing page](imgs/Interface.png)
The interface is split into two main parts, Grammar and Vocab, which in turn offer each a "Browsing" and a "Compare" view.

### Grammar
The grammar browsing view allows the user to input the language, the grammar topic and more fine-grained criteria to find certain grammar rules. Select a specific rule then expands it below to show all details.
![The grammar browsing view](imgs/Grammar_browsing.png)

The compare view is best suited to comparing two or more languages' grammar on certain topics, as illustrated by the comparative in German and Romanian.
![The grammar compare view](imgs/Grammar_compare.png)

### Vocab
Similarly, the vocab browsing view lets the user specify filters to find words of a specific part of speech and, optionally, pertaining to a given topic or derived from a certain language. Clicking a word expands it below.
![The vocab browsing view](imgs/Vocab_browsing.png)

Again, analogously, the vocab compare view serves to compare languages; the only difference to the grammar being that the limit here is two languages.
![The vocab compare view](imgs/Vocab_compare.png)

## Limitations

## Background

### What? - The Idea

The aim of this project is to provide an interface where users can browse and compare the two core components of language learning - grammar and vocabulary - across different languages. Currently, supported languages[^1] include the Romance languages French, Italian, Spanish, Portuguese and Romanian, the Bantu languages Chichewa (Chewa, Nyanja), Chishona (Shona), Kiswahili (Swahili), isiZulu (Zulu).

The interface, split according to the two categories Grammar and Vocabular, allows learners to interact with structured and uniform information sourced from Wikipedia and textbook descriptions of the languages, synthesised by GPT-5.2 on that basis.

### Why? - The Motivation

Learning becomes more efficient when learners can recognize patterns, connect new information to what they already know, and evaluate similarities and differences among concepts. These processes help structure new information, support the formation of meaningful categories, and facilitate the organization of knowledge in a way that allows for prediction and generalization.

Thus, processing information about a new language can benefit from putting it into the context of already known languages to better find similarities and differences. However, traditional learning materials neglect this cross-lingual approach, focussing on the target language only. Furthermore, there is no one description of language, every article or textbook may present and structure the information differently, based on their view of the matter. This is inherently limiting, as it does not allow factoring in how a particular learner best learns.

This is why the author came up with this design, that:
A) presents linguistic data in a uniform and cross-lingual way
B) allows for contextualizing new linguistic information within a wider ecosystem of languages
C) facilitates flexible and dynamic display of the data according to a number of parameters

### How? - The Process

#### Data Preparation

The first step was to bring the Grammar data, sourced from textbook-like word documents[^2] as well as Wikipedia Articles into a style that would allow for GPT-5.2 to efficiently process the data and bring it into the specified format.

For the Vocabulary data, the [Kaikki](https://kaikki.org/) Wiktionary archive was used as is: the wiktionary style of keeping vocabulary records was used as the Gold Standard (even if it is not entirely consistent within itself). Future work could look into augmenting that data with external Vocabulary sources, and even tying these in with the Grammar (e.g., linking inflection patterns to the words that follow it).

The second step was then to specify that format, which was actually a major difficulty and needed multiple passes until a functional schema was devised. This resulted in 18 separate schemas [^3]

The third step consistent in aligning the Grammar schemas with relevant or potentially relevant sections from the source data, as cluttering the prompt with irrelevant information is not conducive to good results. This matching was done by first looking for lexical matches in titles or subtitles of a given article based on a minimal set of relevant search terms. E.g., if the grammar schema is "Determiners" and we are looking for relevant content, the set of keys is {determiner, article, det, art}. As a fallback, embedding of the section text was used (with Sentence Transformer, i.e. Cosine Similarity). If the similarity value fell below, 0.5, the content was not used, lest we steer the LLM wrong.

#### Data Generation

Next, a prompt had to be devised. Inside the prompt was the following content and instructions:
...

GPT-5.2 was then prompted with that prompt and produced the standardised grammar description schemas.

#### Data Postprocessing

The generated schemas then had to be validated and, if necessary, fixed. This was done with ...

[^1]: Note that Grammar descriptions exist also for German. Unlike the others, these were manually done by the author to serve as a Gold Standard of how these grammar descriptions schema should look like. For more information, see [here](#why---the-motivation).
[^2]: ...
[^3]: ...
