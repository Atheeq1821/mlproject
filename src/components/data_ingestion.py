import os
import sys
from src.exceptions import CustomeException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTranformation
@dataclass
class DataIngestionConfig:
    train_data_path=os.path.join('artifacts','train.csv')
    test_data_path=os.path.join('artifacts','test.csv')
    raw_data_path=os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    def initiateDataIngestion(self):
        logging.info("Initiated Dta Ingetion") 
        try:
            df=pd.read_csv('notebook\data\stud.csv')
            logging.info("Dataset loaded")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Training and testing split initialized")

            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Data Ingestion completed")

            return (self.ingestion_config.train_data_path,self.ingestion_config.test_data_path)
        except Exception as e:
            raise CustomeException(e,sys)
if __name__=="__main__":
    obj=DataIngestion()
    train_path,test_path=obj.initiateDataIngestion()
    data_tranform=DataTranformation()
    a,b,c=data_tranform.initiate_transformation(train_path=train_path,test_path=test_path)


            