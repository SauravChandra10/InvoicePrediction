from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import sys

from src.pipeline.prediction_pipeline import PredictPipeline
from src.exception import CustomException

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
        try:
            if 'test_file' not in request.files:
                res={
                    'status' : False,
                    'data' : 'No file part'
                }
                return jsonify(res),400
            
            file = request.files['test_file']

            if file.filename == '':
                res={
                    'status' : False,
                    'data' : 'No selected file'
                }
                return jsonify(res),400
            
            df = pd.read_csv(file)

            predict_pipeline = PredictPipeline()

            pred = predict_pipeline.predict(df)

            res = {
                 'status' : True,
                 'data' : f'Prediction done succesfully, {pred}'
            }

            return jsonify(res)

        except Exception as e:
             raise CustomException(e,sys)
        
if __name__ == '__main__':
     app.run(host = '0.0.0.0', debug = True, port = 5000)