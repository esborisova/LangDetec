"""Script for token collection (using the English spaCy tokenizer regadless the language of a document )"""
import sys

import pandas as pd
import spacy

sys.path.append("../langdetect/")
from preprocessing_text import collect_tokens

nlp = spacy.load(
    "en_core_web_lg",
    disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"],
)
df = pd.read_pickle("../../pkl/dataset.pkl")
docs = df["text"].to_list()
all_tokens = [collect_tokens(doc, nlp) for doc in docs]
df["tokens"] = all_tokens
df = df.drop(["text"], axis=1)
df.to_pickle("../../pkl/tokens.pkl")
