import sys
import os
import pickle
import pandas as pd
from src.exceptions import CustomeException
from src.utils import load_object
from src.logger import logging
class Predict:
    def __init__(self):
        pass
    
    def prediction_output(self,features):
        logging.info("Predicting the output is started .....")
        model_path='artifacts/trained_model.pkl'
        preprocessor_path='artifacts/preprosessing.pkl'

        model=load_object(file_path=model_path)
        preprocessor_obj=load_object(file_path=preprocessor_path)
        logging.info("pickle files gained")
        preprocessed_features=preprocessor_obj.transform(features)
        logging.info("feature transformation")
        predicted_score=model.predict(preprocessed_features)
        logging.info("prediction completed")
        return predicted_score



class Get_Data:
    def __init__(self,gender,race_ethnicity,parental_level_of_education,lunch,
                 test_preparation_course,reading_score,writing_score,math_score):
        self.gender=gender
        self.race_ethnicity=race_ethnicity
        self.parental_level_of_education=parental_level_of_education
        self.lunch=lunch
        self.test_preparation_course=test_preparation_course
        self.reading_score=reading_score
        self.writing_score=writing_score
        self.math_score=math_score

    def user_data_into_dataFrame(self):
        try:
            data_dict={
                'gender':[self.gender],
                'race_ethnicity':[self.race_ethnicity],
                'parental_level_of_education':[self.parental_level_of_education],
                'lunch':[self.lunch],
                'test_preparation_course':[self.test_preparation_course],
                'reading_score':[self.reading_score],
                'writing_score':[self.writing_score],
                'math_score':[self.math_score]
            } 
            logging.info("user data successfully converted into dataframe")
            return pd.DataFrame(data_dict)   
        except Exception as e:
            raise CustomeException(e,sys)

        