import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

from diagnostics import model_predictions


###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

model_path = os.path.join(config['output_model_path'])
test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])


##############Function for reporting
def score_model(prod_deployment_path:str, test_data_path:str, model_path:str, img_name:str="confusionmatrix.png"):
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace
    predictions = model_predictions(prod_deployment_path=prod_deployment_path, data_file=os.path.join(test_data_path,"testdata.csv"))
    df = pd.read_csv(os.path.join(test_data_path,"testdata.csv"))
    y_test = df["exited"].values
    cm = metrics.confusion_matrix(y_test, predictions)
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(y_test)))
    plt.xticks(tick_marks, ['Predicted Negative', 'Predicted Positive'])
    plt.yticks(tick_marks, ['Actual Negative', 'Actual Positive'])
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    plt.tight_layout()
    plt.savefig(os.path.join(model_path,img_name))


if __name__ == '__main__':
    score_model(
        prod_deployment_path=prod_deployment_path,
        test_data_path=test_data_path,
        model_path=model_path
    )
