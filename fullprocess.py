import json
import os
import pickle

import training
import scoring
import deployment
import apicalls
import reporting
import ingestion


with open('config.json','r') as f:
    config = json.load(f)

input_folder_path = os.path.join(config['input_folder_path'])
output_folder_path = os.path.join(config['output_folder_path'])
output_model_path = os.path.join(config['output_model_path'])
test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])


def automatically_process(input_folder_path:str,
                          output_folder_path:str,
                          output_model_path:str,
                          test_data_path:str,
                          prod_deployment_path:str):
    ##################Check and read new data
    #first, read ingestedfiles.txt
    print("Reading ingested data ...")
    with open(os.path.join(prod_deployment_path, "ingestedfiles.txt"),"r") as f:
        prev_file_list = f.readlines()
        prev_file_list = [file.strip() for file in prev_file_list]

    #second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
    print("Reading new data ...")
    current_file_list = os.listdir(input_folder_path)
    check_new_data = False
    for f in current_file_list:
        check_new_data = check_new_data or (f not in prev_file_list)

    ##################Deciding whether to proceed, part 1
    #if you found new data, you should proceed. otherwise, do end the process here
    if check_new_data:
        print("Ingest new data ...")
        ingestion.merge_multiple_dataframe(
            input_folder_path=input_folder_path,
            output_folder_path=output_folder_path
        )
    else:
        return "No new data"

    ##################Checking for model drift
    #check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
    print("Checking model drift ...")
    with open(os.path.join(prod_deployment_path, "latestscore.txt"),"r") as f:
        prev_score = float(f.read())
    new_score = scoring.score_model(
        model_path=prod_deployment_path,
        data_file=os.path.join(output_folder_path, "finaldata.csv"),
        write_score=False
    )
    print(f"new score = {new_score}")
    check_model_drift = new_score < prev_score
    check_model_drift = True

    ##################Deciding whether to proceed, part 2
    #if you found model drift, you should proceed. otherwise, do end the process here
    if not check_model_drift:
        return "No model drift"
    else:
        print("Re-train new model")
        training.train_model(
            dataset_csv_path=output_folder_path,
            model_path=output_model_path,
        )
        scoring.score_model(
            data_file=os.path.join(test_data_path, "testdata.csv"),
            model_path=output_model_path,
            write_score=True,
        )

    ##################Re-deployment
    #if you found evidence for model drift, re-run the deployment.py script
        print("Re-deploy new model")
        deployment.store_model_into_pickle(
            model_path=output_model_path,
            dataset_csv_path=output_folder_path,
            prod_deployment_path=prod_deployment_path
        )

    ##################Diagnostics and reporting
    #run diagnostics.py and reporting.py for the re-deployed model
        print("Re-diagnostics ...")
        apicalls.diagnostics(
            out_file_name="apireturns2.txt"
        )
        print("Reporting ...")
        reporting.score_model(
            prod_deployment_path=prod_deployment_path,
            test_data_path=test_data_path,
            model_path=output_model_path,
            img_name="confusionmatrix2.png"
        )
    return "Deployed new model"


if __name__ == "__main__":
    ans = automatically_process(
        input_folder_path=input_folder_path,
        output_folder_path=output_folder_path,
        output_model_path=output_model_path,
        test_data_path=test_data_path,
        prod_deployment_path=prod_deployment_path
    )
    print(ans)




