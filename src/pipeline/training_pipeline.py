import os
import sys 
from src.exception import CustomException
from src.logger import logging

class TrainPipeline:
    def __init__(self,df):
        self.df = df
        self.data_path:str = os.path.join('artifacts','data.csv')
        self.preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

    def initiate_data_ingestion(self):
        logging.info("Initiated data ingestion")

        try:            
            os.makedirs(os.path.dirname(self.data_path),exist_ok=True)
            self.df.to_csv(self.data_path,index=False,header=True)

            logging.info("Data ingestion is complete")

            return self.data_path

        except Exception as e:
            raise CustomException(e,sys)
        