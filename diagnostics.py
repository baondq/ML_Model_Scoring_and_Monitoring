
import pandas as pd
import numpy as np
import pickle
import timeit
import os
import json
import subprocess

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

prod_deployment_path = os.path.join(config['prod_deployment_path']) 
test_data_path = os.path.join(config['test_data_path'])
dataset_csv_path = os.path.join(config['output_folder_path'])

##################Function to get model predictions
def model_predictions(prod_deployment_path:str, data_file:str):
    #read the deployed model and a test dataset, calculate predictions
    with open(os.path.join(prod_deployment_path, "trainedmodel.pkl"), "rb") as f:
        model = pickle.load(f)
    df = pd.read_csv(data_file)
    X_test = df[["lastmonth_activity","lastyear_activity","number_of_employees"]].values
    predictions = model.predict(X_test)
    return predictions.tolist()  #return value should be a list containing all predictions

##################Function to get summary statistics
def dataframe_summary(test_data_path:str):
    #calculate summary statistics here
    df = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))
    df = df.select_dtypes(include="number")
    mean_values = df.mean().values.tolist()
    median_values = df.median().values.tolist()
    std_values = df.std().values.tolist()
    return [[mean_values[i], median_values[i], std_values[i]] for i in range(len(mean_values))] #return value should be a list containing all summary statistics

##################Function to check missing data
def missing_data(dataset_csv_path:str):
    final_df = pd.read_csv(os.path.join(dataset_csv_path, "finaldata.csv"))
    missing_values_df = final_df.isna().sum() / final_df.shape[0]
    return missing_values_df.values.tolist()

##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    time_ingestion = timeit.timeit(stmt="merge_multiple_dataframe()", setup="from ingestion import merge_multiple_dataframe", number=1)

    time_training = timeit.timeit(stmt="train_model()", setup="from training import train_model", number=1)
    
    return [time_ingestion, time_training]   #return a list of 2 timing values in seconds

##################Function to check dependencies
def outdated_packages_list():
    with open("requirements.txt","r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    package_list = []
    current_version_list = []
    most_recent_version_list = []
    for line in lines:
        package_list.append(line.split("==")[0])
        current_version_list.append(line.split("==")[1])
        try:
            latest_version = subprocess.check_output(["pip", "index", "versions", package_list[-1]]).decode().split("\n")[-2].split(":")[1].strip()
            most_recent_version_list.append(latest_version)
        except subprocess.CalledProcessError:
            most_recent_version_list.append("Unknown")
    df = pd.DataFrame({
        "package": package_list,
        "current version": current_version_list,
        "most recent version": most_recent_version_list
    })
    return df


if __name__ == '__main__':
    predictions = model_predictions(
        prod_deployment_path=prod_deployment_path,
        data_file=os.path.join(test_data_path,"testdata.csv")
    )
    summary = dataframe_summary(
        test_data_path=test_data_path
    )
    missing = missing_data(
        dataset_csv_path=dataset_csv_path
    )
    t = execution_time()
    packages = outdated_packages_list()

    print(predictions)
    print(summary)
    print(missing)
    print(t)
    print(packages)




    
