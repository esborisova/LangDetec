import sys

sys.path.append("../src/langdetect")
from preprocessing_text import collect_tokens
import pytest
import pandas as pd
import datatest as dt
import spacy


def test_collect_tokens():

    nlp = spacy.load(
        "en_core_web_lg",
        disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"],
    )
    text = "madam president i should like to draw your attention to a case in which this parliament has consistently shown an interest"
    tokens = collect_tokens(text, nlp)

    assert isinstance(tokens, list)
    assert tokens == [
        "madam",
        "president",
        "i",
        "should",
        "like",
        "to",
        "draw",
        "your",
        "attention",
        "to",
        "a",
        "case",
        "in",
        "which",
        "this",
        "parliament",
        "has",
        "consistently",
        "shown",
        "an",
        "interest",
    ]
