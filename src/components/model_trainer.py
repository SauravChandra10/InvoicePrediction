import os
import sys
import numpy as np

from dataclasses import dataclass
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression

from src.utils import save_object, evaluate_models
from src.logger import logging
from src.exception import CustomException

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            X_train,y_train,X_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models = {
                "Gradient Boosting Regressor" : GradientBoostingRegressor(),
                "Linear Regression" : LinearRegression()
            }

            params = {
                "Gradient Boosting Regressor":{
                    "learning_rate" : [0.001],
                    "max_depth" : [8],
                    "n_estimators" : [2000],
                    "subsample" : [0.5]
                },
                "Linear Regression":{}
            }

            model_report:dict = evaluate_models(
                X_train,y_train,X_test,y_test,models,params
            )

            best_score = min(model_report.values())

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_score)]

            best_model = models[best_model_name]

            logging.info(f"{best_model_name}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            return best_model_name,best_score

        except Exception as e:
            raise CustomException(e,sys)