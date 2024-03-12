import sys 
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,df):

        try:

            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'

            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)

            # remove duplicates
            df.drop_duplicates(inplace=True)

            # remove NaN

            # covert date into day,month and year
            df['Invoice Date'] = pd.to_datetime(df['Invoice Date'], format='%d-%m-%Y')
            df['Day'] = df['Invoice Date'].dt.day
            df['Month'] = df['Invoice Date'].dt.month
            df['Year'] = df['Invoice Date'].dt.year

            df.drop(columns=['Invoice Date'],inplace=True)

            data_scaled = preprocessor.transform(df)

            pred = model.predict(data_scaled)

            return pred
        
        except Exception as e:
            raise CustomException(e,sys)