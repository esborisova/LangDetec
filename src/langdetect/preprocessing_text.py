"""Functions for processing the text documents (cleaning)"""
from bs4 import BeautifulSoup
import regex as re
from typing import List


def remove_html_commands(text: str) -> str:
    """
    Cleans text from html tags.
    Args:
        text (str): The string to remove tags from.
    Returns:
        str: A string cleaned from html tags.
    """
    soup = BeautifulSoup(text, features="html.parser")

    for script in soup(["script", "style"]):
        script.extract()

    text = soup.get_text()

    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = "\n".join(chunk for chunk in chunks if chunk)

    return text


def clean_text(text: str) -> str:
    """
    Cleans text from punctuation, URLs, ASCII charecters (except special letters, f.ex., German umlaut),
    multiple spaces and lowercases.
    Args:
        text (str): The text document to clean.
    Returns:
        str: The cleaned document.
    """
    no_urls = re.sub(r"http\S+", "", text)
    no_special_ch = re.sub(r"([^A-Za-zØøÅåÆæÄäÖöÜüẞß])|(\w+:\/\/\S+)", " ", no_urls)
    no_special_ch = no_special_ch.replace("\n", " ")
    lowercased_str = no_special_ch.lower()
    cleaned_text = " ".join(lowercased_str.split())

    return cleaned_text
