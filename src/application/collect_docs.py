"""Pipeline for collecting English, Danish, German snd Swedish texts, 
cleaning and saving into separate json files"""
import sys

sys.path.append("../langdetect/")
import pandas as pd
from load_data import get_files
from preprocessing_text import remove_html_commands, clean_text
import json


root_dir = "../../data/"
paths = get_files(root_dir)

dataset = []

for file in paths:
    if str(sys.argv[1]) in file:
        with open(file, encoding="utf-8") as f:
            text_label = {}
            data = f.readlines()
            data = " ".join(data)
            data = data[:2000]
            cleaned_doc = remove_html_commands(data)
            cleaned = clean_text(cleaned_doc)
            text_label["text"] = cleaned
            text_label["language"] = str(sys.argv[1])
    else:
        continue
    dataset.append(text_label)

    if len(dataset) == 10000:
        break


data = json.dumps(dataset, ensure_ascii=False, indent=2)
file_name = "../../json/" + str(sys.argv[1]) + ".json"

with open(file_name, "w") as outfile:
    outfile.write(data)
