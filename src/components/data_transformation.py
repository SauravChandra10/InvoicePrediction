import os
import sys 
import numpy as np
import pandas as pd

from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            categorical_columns = [
                'Invoice Currency'
            ]
            numerical_columns   = [
                'Customer Name',
                'Credit Terms',
                'Invoice Amount',
                'Day',
                'Month',
                'Year'
            ]

            num_pipeline = Pipeline(
                steps = [
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )

            logging.info("Pipeline has been completed")

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_columns),
                    ('cat_pipeline',cat_pipeline,categorical_columns)
                ]
            )

            logging.info("Preprocessing done")

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,data_path):
        try:
            df = pd.read_csv(data_path)
            logging.info("Read data into dataframe")

            # remove duplicates
            df.drop_duplicates(inplace=True)

            # remove NaN
            df = df.dropna(subset=['Actual Delay over and above Agreed Credit Terms'])

            # covert date into day,month and year
            df['Invoice Date '] = pd.to_datetime(df['Invoice Date '], format='%Y-%m-%d')
            df['Day'] = df['Invoice Date '].dt.day
            df['Month'] = df['Invoice Date '].dt.month
            df['Year'] = df['Invoice Date '].dt.year

            df.drop(columns=['Invoice Date '],inplace=True)

            X=df.drop(['Actual Delay over and above Agreed Credit Terms'],axis=1)
            y=df['Actual Delay over and above Agreed Credit Terms']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

            preprocessing_obj = self.get_data_transformer_object()

            X_train_arr = preprocessing_obj.fit_transform(X_train)
            X_test_arr = preprocessing_obj.transform(X_test)

            train_arr = np.c_[
                X_train_arr,np.array(y_train)
            ]

            test_arr = np.c_[
                X_test_arr,np.array(y_test)
            ]

            logging.info("Saved preprocessing object")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            raise CustomException(e,sys)
