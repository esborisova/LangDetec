"""Script for creating a dataframe from json files"""
import json
import pandas as pd
import os


dfs = []

root_dir = "../../json/"
file_list = ["en.json", "sv.json", "de.json", "da.json"]

for file in file_list:
    file_path = os.path.join(root_dir, file)
    with open(file_path) as f:
        json_data = pd.json_normalize(json.load(f))
        json_data = json_data.drop_duplicates(
            subset=["text"], keep="first"
        ).reset_index(drop=True)
    dfs.append(json_data)
df = pd.concat(dfs, sort=False)
df.to_pickle("../../pkl/dataset.pkl")
