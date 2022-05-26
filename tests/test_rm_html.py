import sys

sys.path.append("../src/langdetect")
from preprocessing_text import remove_html_commands
import pytest
import pandas as pd
import datatest as dt


def test_remove_html_commands():
    sample = "<SPEAKER ID=54 NAME=Der Präsident>Die Aussprache ist geschlossen.<P>"
    cleaned = remove_html_commands(sample)

    assert isinstance(cleaned, str)
    assert cleaned == "Die Aussprache ist geschlossen."
