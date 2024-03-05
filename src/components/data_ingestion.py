import os, sys
from src.exception import CustomException
import pandas as pd
from src.logger import logging
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    data_path:str = os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Initiated data ingestion")

        try:
            df = pd.read_csv('raw.csv')
            
            logging.info("Read raw.csv as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.data_path,index=False,header=True)

            logging.info("Data ingestion is complete")

            return self.ingestion_config.data_path

        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()