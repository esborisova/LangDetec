"""Script for creating a dataframe from json files"""
import json
import pandas as pd
import os


dfs = []

root_dir = "../../json/"
file_list = ["en.json", "sv.json", "de.json", "da.json", "it.json"]

for file in file_list:
    file_path = os.path.join(root_dir, file)
    with open(file_path) as f:
        df = pd.json_normalize(json.load(f))
        df = df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
        df = df[df["text"] != ""]
    dfs.append(df)
data_df = pd.concat(dfs, sort=False)
duplicated = data_df[data_df.duplicated()]
print(duplicated)
data_df.to_pickle("../../pkl/dataset.pkl")
