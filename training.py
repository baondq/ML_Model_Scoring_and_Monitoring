from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path'])

os.makedirs(model_path, exist_ok=True)


#################Function for training the model
def train_model(dataset_csv_path:str=dataset_csv_path, model_path:str=model_path):
    # load data
    df = pd.read_csv(os.path.join(dataset_csv_path, "finaldata.csv"))
    X_train = df[["lastmonth_activity","lastyear_activity","number_of_employees"]].values
    y_train = df["exited"].values
    
    #use this logistic regression for training
    model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='ovr', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
    
    #fit the logistic regression to your data
    model.fit(X_train, y_train)
    
    #write the trained model to your workspace in a file called trainedmodel.pkl
    with open(os.path.join(model_path, "trainedmodel.pkl"), "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    train_model(
        dataset_csv_path=dataset_csv_path,
        model_path=model_path
    )
