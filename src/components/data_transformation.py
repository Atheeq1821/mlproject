import pandas as pd
import os
import sys
import numpy as np
from src.logger import logging
from src.exceptions import CustomeException
from dataclasses import dataclass
from src.utils import save_object
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder

from sklearn.compose import ColumnTransformer

@dataclass
class DataTranformationConfig:
    data_transformation_pickle_path=os.path.join('artifacts',"preprosessing.pkl")


class DataTranformation:
    def __init__(self):
        self.data_tranformationConfig = DataTranformationConfig()
    
    def get_data_tranformer(self):
        """ This function tranforms data preprocess it and returns the preprocessor object"""

        try:
            num_columns=['reading_score','writing_score','math_score']
            cat_columns=[
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
                ]
            
            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )
            logging.info(f'Numerical columns are {num_columns}')

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('onehotencoding',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )

            logging.info(f'Categorical columns are {cat_columns}')

            preprocessor= ColumnTransformer(
                [
                    ('numerical_pipeline',num_pipeline,num_columns),
                    ("categorical_pipeline",cat_pipeline,cat_columns)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomeException(e,sys)
    def initiate_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("data read for preprocessing")

            target_column='average'
            transformer_obj=self.get_data_tranformer()

            logging.info("Created transformaer object")

            train_features=train_df.drop(columns=[target_column],axis=1)
            test_features=test_df.drop(columns=[target_column],axis=1)
            train_target=train_df[target_column]
            test_target=test_df[target_column]

            logging.info("features and target separated form train and test data")

            train_X=transformer_obj.fit_transform(train_features)
            test_X=transformer_obj.transform(test_features)

            logging.info("fit_tranform on train features and tranform on test features")

            train_complete = np.c_[
                    train_X, np.array(train_target)
                ]

            test_complete = np.c_[
                    test_X, np.array(test_target)
                ]
            
            filepath=self.data_tranformationConfig.data_transformation_pickle_path
            logging.info('saving.... data_tranformer.plk file')

            save_object(
                file_path=filepath,
                obj=transformer_obj
            )
            return (
                train_complete,test_complete,filepath
            )

        except Exception as e:
            raise CustomeException(e,sys)
    