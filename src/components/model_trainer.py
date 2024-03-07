import os
import sys
import numpy as np

from dataclasses import dataclass
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.utils import save_object
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
            logging.info('Spliting train and test arr')

            X_train,y_train,X_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            model = GradientBoostingRegressor(
                learning_rate = 0.001,
                max_depth = 8,
                n_estimators = 2000,
                subsample = 0.5
            )

            model.fit(X_train,y_train)

            pred = model.predict(X_test)

            r2 = r2_score(y_test,pred)
            mae = mean_absolute_error(y_test,pred)
            rmse = np.sqrt(mean_squared_error(y_test,pred))

            logging.info(f"r2:{r2}, mae:{mae}, rmse:{rmse}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = model
            )

        except Exception as e:
            raise CustomException(e,sys)