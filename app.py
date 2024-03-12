from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import sys

from src.pipeline.prediction_pipeline import PredictPipeline
from src.exception import CustomException

app = Flask(__name__)

@app.errorhandler(CustomException)
def handle_my_error(error):
    res={
        "status":False,
        "message":"Error!",
        "data":error
    }
    response = jsonify(res)
    return response

@app.route('/invoice')
def index():
    return render_template('index.html')

@app.route('/invoicepredict', methods=['POST'])
def predict():
        try:
            if request.is_json:
                return predictJSON()
            
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
            error = CustomException(e,sys).error_message
            return handle_my_error(error)
        
def predictJSON():
        try:
            data = request.json

            if not data:
                res={
                    'status' : False,
                    'data' : 'Invalid JSON data'
                }
                return jsonify(res),400
            
            required_keys = ['Customer Name', 'Invoice Date', 'Credit Terms', 
                            'Invoice Currency', 'Invoice Amount']
            
            missing_keys = [key for key in required_keys if key not in data]

            if missing_keys:
                res = {
                    'status': False,
                    'data': f'Missing keys in JSON data: {missing_keys}'
                }
                return jsonify(res), 400
        
            df = pd.DataFrame([data])

            predict_pipeline = PredictPipeline()

            pred = predict_pipeline.predict(df)

            res={
                'status' : True,
                'data' : f'Prediction done succesfully, {pred}'
            }
            return jsonify(res)

        except Exception as e:
            error = CustomException(e,sys).error_message
            return handle_my_error(error)
        
if __name__ == '__main__':
     app.run(host = '0.0.0.0', debug = True, port = 5000)