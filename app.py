from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import json
import os

from scoring import score_model
from diagnostics import model_predictions, dataframe_summary, missing_data, execution_time, outdated_packages_list

######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path'])
test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])


#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    #call the prediction function you created in Step 3
    filename = request.args.get('input_data')
    if os.path.isfile(filename) is False:
        return f"{filename} doesn't exist"
    pred = model_predictions(
        prod_deployment_path=prod_deployment_path,
        data_file=filename
    )
    return ",".join(str(p) for p in pred), 200 #add return value for prediction outputs

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def score():        
    #check the score of the deployed model
    f1score = score_model(
        data_file=os.path.join(test_data_path, "testdata.csv"),
        model_path=prod_deployment_path
    )
    return jsonify(f1score), 200 #add return value (a single F1 score number)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def summary():        
    #check means, medians, and modes for each column
    statistics = dataframe_summary(test_data_path=test_data_path)
    return jsonify(statistics), 200 #return a list of all calculated summary statistics

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():        
    #check timing and percent NA values
    timing = execution_time()
    missing = missing_data(dataset_csv_path=dataset_csv_path)
    dependencies = outdated_packages_list()
    res = {
        'timing': timing,
        'missing_data': missing,
        'dependency_check': dependencies.to_json(orient="records", lines=True),
    }
    return jsonify(res), 200#add return value for all diagnostics

if __name__ == "__main__":    
    app.run(host='127.0.0.1', port=5000, debug=True, threaded=True)
