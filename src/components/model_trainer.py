import pandas as pd
import numpy as numpy
import sys
import os
from dataclasses import dataclass

from src.exceptions import CustomeException
from src.logger import logging
from src.utils import evaluate_model,save_object
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

@dataclass
class ModelTrainerConfig:
    model_trained_path=os.path.join('artifacts','trained_model.pkl')

class ModelTrainer:
    def __init__(self):
        self.trained_model_file_path=ModelTrainerConfig()

    def training_model(self,train_data,test_data):
        try:
            logging.info("data split started for model training")
            x_train=train_data[:,:-1]
            y_train=train_data[:,-1]
            x_test=test_data[:,:-1]
            y_test=test_data[:,-1]
            models={
                "LinearRegressor":LinearRegression(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "RandomForestRegressor": RandomForestRegressor(),
                "GradientBoostingRegressor":GradientBoostingRegressor(),
                "AdaBoostRegressor": AdaBoostRegressor(),
                "XGBRegressor":XGBRegressor(),
                "CatBoostRegressor":CatBoostRegressor()

            }
            logging.info("Model training started")
            best_model_name,best_accuracy=evaluate_model(x_train,y_train,x_test,y_test,models)
            if best_accuracy<0.6:
                raise CustomeException("No best model found",sys)
            
            logging.info("trainer file saving....")

            save_object(
                file_path=self.trained_model_file_path.model_trained_path,
                obj=models[best_model_name]
            )
            logging.info(f"Best model found with accuracy {best_accuracy} and the model is {best_model_name}")

            return best_accuracy
        except Exception as e:
            raise CustomeException(e,sys)    

