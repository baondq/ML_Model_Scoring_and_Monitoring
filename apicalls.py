import requests
import os
import json


#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:5000/"

def diagnostics(out_file_name:str="apireturns.txt"):
#Call each API endpoint and store the responses
    response1 = requests.post(URL + '/prediction?input_data=testdata/testdata.csv').content.decode("utf-8") #put an API call here
    response2 = requests.get(URL + '/scoring').content.decode("utf-8") #put an API call here
    response3 = requests.get(URL + '/summarystats').content.decode("utf-8") #put an API call here
    response4 = requests.get(URL + '/diagnostics').content.decode("utf-8") #put an API call here

#combine all API responses
    responses = {
        "prediction": response1,
        "scoring": response2,
        "summarystats": response3,
        "diagnostics": response4,
    } #combine reponses here

#write the responses to your workspace
    with open('config.json','r') as f:
        config = json.load(f)

    model_path = os.path.join(config['output_model_path'])

    with open(os.path.join(model_path, out_file_name), "w") as f:
        f.write(json.dumps(responses, indent=2))


if __name__ == "__main__":
    diagnostics()
