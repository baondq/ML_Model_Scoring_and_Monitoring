from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

model_path = os.path.join(config['output_model_path']) 
test_data_path = os.path.join(config['test_data_path']) 


#################Function for model scoring
def score_model(data_file:str, model_path:str, write_score:bool=True):
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    # load test data
    df = pd.read_csv(data_file)
    X_test = df[["lastmonth_activity","lastyear_activity","number_of_employees"]].values
    y_test = df["exited"].values
    # Load the trained model
    with open(os.path.join(model_path, "trainedmodel.pkl"), "rb") as f:
        model = pickle.load(f)
    # Calculate the F1 score of the trained model on the test data
    y_pred = model.predict(X_test)
    f1 = metrics.f1_score(y_test, y_pred)
    if write_score:
        with open(os.path.join(model_path, "latestscore.txt"), "w") as f:
            f.write(str(f1.item()))
    return f1


if __name__ == "__main__":
    score_model(
        data_file=os.path.join(test_data_path, "testdata.csv"),
        model_path=model_path
    )
