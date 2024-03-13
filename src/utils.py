import pandas as pd
import numpy as np
import sys
import os
from src.exceptions import CustomeException
from src.logger import logging
import pickle

from sklearn.metrics import r2_score
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomeException(e, sys)

def evaluate_model(x_train,y_train,x_test,y_test,models):
    try:
        best_model_name=""
        best_accuracy=0

        for i in models.keys():
            model=models[i]
            model.fit(x_train,y_train)
            pred_y=model.predict(x_test)
            score=r2_score(y_test,pred_y)
            if score>best_accuracy:
                best_accuracy=score
                best_model_name=i                
        logging.info("Model training completed")      

        return (best_model_name,best_accuracy)   
         
    except Exception as e:
        raise CustomeException(e,sys)