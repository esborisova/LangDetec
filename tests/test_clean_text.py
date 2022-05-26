import sys

sys.path.append("../src/langdetect")
from preprocessing_text import clean_text
import pytest
import pandas as pd
import datatest as dt


def test_clean_text():

    sample = "Evans, jeg mener,     at et initiativ,   som det, De foreslår; ville være meget hensigtsmæssigt https://en.wikipedia.org/wiki/Dot_product 1111."
    cleaned = clean_text(sample)

    assert isinstance(cleaned, str)
    assert (
        cleaned
        == "evans jeg mener at et initiativ som det de foreslår ville være meget hensigtsmæssigt"
    )
