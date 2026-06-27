# Language Identification with Character N-grams

This project demonstrates how to identify the language of a piece of text using
statistical character n-gram models. The implementation is contained in a
Jupyter notebook and supports four languages:

- English
- Spanish
- Czech
- Polish

Rather than relying on a pre-trained classifier, the notebook builds one
language model per language from a collection of public-domain books. An input
text is scored by every model, and the language whose character patterns assign
it the highest probability is returned as the best match.

## What the notebook does

The notebook implements the complete experiment from data preparation through
language prediction:

1. Downloads four UTF-8 corpora packaged as ZIP archives. The archives contain
   books sourced from [Project Gutenberg](https://www.gutenberg.org/).
2. Merges the text files for each language and tokenizes them with the
   language-specific `MosesTokenizer` from `sacremoses`.
3. Splits each corpus sequentially into 80% training, 10% held-out, and 10% test
   data.
4. Counts character unigrams, bigrams, and trigrams for each language. Bigram
   and trigram probabilities are estimated conditionally from their preceding
   characters.
5. Tunes an additive smoothing value for each trigram model on the held-out
   data. Candidate values from `0.01` through `0.99` are compared by cross-entropy.
6. Evaluates every language model against every test corpus.
7. Provides `identify_language(text)`, which tokenizes arbitrary input, scores
   it with all four smoothed trigram models, and returns ranked
   `(probability, language)` pairs.

The notebook also reports corpus sizes, the five most frequent character
trigrams per language, and out-of-vocabulary word percentages for the held-out
and test sets.

## Results

The selected additive smoothing parameters were:

| Language | Smoothing value |
| --- | ---: |
| English | 0.01 |
| Spanish | 0.04 |
| Czech | 0.04 |
| Polish | 0.02 |

Cross-entropy in bits per character is shown below. Lower values indicate that
the model considers the test text more likely.

| Model \ Test corpus | English | Spanish | Czech | Polish |
| --- | ---: | ---: | ---: | ---: |
| English | **2.825** | 4.607 | 6.851 | 7.009 |
| Spanish | 4.483 | **3.748** | 6.326 | 6.934 |
| Czech | 4.656 | 5.255 | **3.259** | 6.034 |
| Polish | 4.432 | 4.878 | 5.796 | **3.302** |

Each test corpus receives its lowest cross-entropy from the matching language
model, showing that the learned character trigram distributions separate all
four languages in this dataset.

For example, the notebook ranks an English sample as follows:

```text
[(0.1480, 'English'),
 (0.0732, 'Spanish'),
 (0.0708, 'Polish'),
 (0.0593, 'Czech')]
```

These values are normalized per-character likelihood scores used for ranking;
they are not calibrated class probabilities and therefore do not sum to one.

## Running the project

Requirements:

- Python 3
- Jupyter Notebook or JupyterLab
- Internet access for the corpus downloads
- `wget`, as used by the notebook's download cells

Open [`src/language_identification.ipynb`](src/language_identification.ipynb)
and run its cells from top to bottom. The notebook installs `sacremoses`, creates
a local `data/` directory, downloads the corpora, trains the models, and prints
the evaluation results.

Because the trained models live in notebook memory, the data preparation and
training cells must be run before calling `identify_language(text)`.

## Project structure

```text
.
├── README.md
└── src/
    └── language_identification.ipynb
```

This is an educational notebook rather than a packaged command-line tool or
service. Its main purpose is to illustrate corpus preparation, character-level
language modeling, smoothing, cross-entropy evaluation, and probability-based
language identification without machine-learning frameworks.
