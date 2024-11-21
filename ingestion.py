import pandas as pd
import numpy as np
import os
import json
from datetime import datetime


#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']

os.makedirs(output_folder_path, exist_ok=True)


#############Function for data ingestion
def merge_multiple_dataframe(input_folder_path:str=input_folder_path, output_folder_path:str=output_folder_path):
    #check for datasets, compile them together, and write to an output file
    file_list = os.listdir(input_folder_path)
    df_list = []
    record_text_list = []
    for f in file_list:
        record_text_list.append(os.path.join(input_folder_path, f))
        df_list.append(pd.read_csv(os.path.join(input_folder_path, f)))
    df = pd.concat(df_list, axis=0)
    df = df.drop_duplicates()
    df.to_csv(os.path.join(output_folder_path, "finaldata.csv"), index=False)
    with open(os.path.join(output_folder_path, "ingestedfiles.txt"), "w") as f:
        f.write("\n".join(record_text_list))


if __name__ == '__main__':
    merge_multiple_dataframe(
        input_folder_path=input_folder_path,
        output_folder_path=output_folder_path
    )
