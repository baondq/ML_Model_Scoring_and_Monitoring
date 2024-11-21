from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import shutil



##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path'])
model_path = os.path.join(config['output_model_path'])

os.makedirs(prod_deployment_path, exist_ok=True)

####################function for deployment
def store_model_into_pickle(model_path:str, dataset_csv_path:str, prod_deployment_path:str):
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    shutil.copy(os.path.join(model_path, "trainedmodel.pkl"), os.path.join(prod_deployment_path, "trainedmodel.pkl"))
    shutil.copy(os.path.join(model_path, 'latestscore.txt'), os.path.join(prod_deployment_path, "latestscore.txt"))
    shutil.copy(os.path.join(dataset_csv_path, 'ingestedfiles.txt'), os.path.join(prod_deployment_path, "ingestedfiles.txt"))
        
        
if __name__ == "__main__":
    store_model_into_pickle(
        model_path=model_path,
        dataset_csv_path=dataset_csv_path,
        prod_deployment_path=prod_deployment_path
    )
