from flask import Flask, request, render_template, jsonify
import pandas as pd
import sys

from src.pipeline.prediction_pipeline import PredictPipeline
from src.pipeline.training_pipeline import TrainPipeline
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

from src.exception import CustomException

app = Flask(__name__)

@app.errorhandler(CustomException)
def handle_my_error(error):
    res={
        "status":False,
        "message":"Error",
        "data":error
    }
    response = jsonify(res)
    return response

@app.route('/invoice')
def index():
    return render_template('index.html')

@app.route('/invoiceprediction', methods=['POST'])
def predict():
        try:
            if request.is_json:
                return predictJSON()
            
            if 'test_file' not in request.files:
                res={
                    'status' : False,
                    'message' : 'Error',
                    'data' : 'No file part'
                }
                return jsonify(res),400
            
            file = request.files['test_file']

            if file.filename == '':
                res={
                    'status' : False,
                    'message' : 'Error',
                    'data' : 'No selected file'
                }
                return jsonify(res),400
            
            df = pd.read_csv(file)

            predict_pipeline = PredictPipeline()

            pred = predict_pipeline.predict(df)

            prediction_list = pred.tolist()

            res = {
                 'status' : True,
                 'message' : 'Prediction done successfully',
                 'data' : prediction_list
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
                    'message' : 'Error',
                    'data' : 'Invalid JSON data'
                }
                return jsonify(res),400
            
            required_keys = ['Customer Name', 'Invoice Date', 'Credit Terms', 
                            'Invoice Currency', 'Invoice Amount']
            
            missing_keys = [key for key in required_keys if key not in data]

            if missing_keys:
                res = {
                    'status': False,
                    'message' : 'Error',
                    'data': f'Missing keys in JSON data: {missing_keys}'
                }
                return jsonify(res), 400
        
            df = pd.DataFrame([data])

            predict_pipeline = PredictPipeline()

            pred = predict_pipeline.predict(df)

            prediction_list = pred.tolist()

            res={
                'status' : True,
                'message' : 'Prediction done successfully',
                'data' : prediction_list
            }
            
            return jsonify(res)

        except Exception as e:
            error = CustomException(e,sys).error_message
            return handle_my_error(error)
        
@app.route('/invoicetraining', methods=['POST'])
def train():
        try:
            if request.is_json:
                return trainJSON()

            if 'train_file' not in request.files:
                res={
                    'status' : False,
                    'message' : 'Error',
                    'data' : 'No file part'
                }
                return jsonify(res),400
            
            file = request.files['train_file']

            if file.filename == '':
                res={
                    'status' : False,
                    'message' : 'Error',
                    'data' : 'No selected file'
                }
                return jsonify(res),400
            
            df = pd.read_excel(file)

            obj = TrainPipeline(df)
            data_path = obj.initiate_data_ingestion()

            data_transformation = DataTransformation()
            train_arr, test_arr, _ = data_transformation.initiate_data_transformation(data_path)

            model_trainer = ModelTrainer()
            model_name, model_score = model_trainer.initiate_model_trainer(train_arr,test_arr)

            res={
                "status":True,
                "message": "Training done successfully",
                "data": f"model created at artifacts/model.pkl, best model is {model_name} and rmse is {model_score}"
            }

            return jsonify(res)


        except Exception as e:
            error = CustomException(e,sys).error_message
            return handle_my_error(error) 

def trainJSON():
    try:
        data = request.json

        if not data:
            res={
                'status' : False,
                'message' : 'Error',
                'data' : 'Invalid JSON data'
            }
            return jsonify(res),400  
        
        required_keys = ['Customer Name', 'Invoice Date', 'Credit Terms', 
                        'Invoice Currency', 'Invoice Amount',
                        'Actual Delay over and above Agreed Credit Terms']
            
        missing_keys = [key for key in required_keys if key not in data]

        if missing_keys:
            res = {
                'status': False,
                'message' : 'Error',
                'data': f'Missing keys in JSON data: {missing_keys}'
            }
            return jsonify(res), 400
        
        df = pd.DataFrame([data])
        obj = TrainPipeline(df)
        data_path = obj.initiate_data_ingestion()

        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(data_path)

        model_trainer = ModelTrainer()
        model_trainer.initiate_model_trainer(train_arr,test_arr)

        res={
            "status":True,
            "message":"Training done successfully",
            "data":"model created at artifacts/model.pkl"
        }

        return jsonify(res)
              
    except Exception as e:
        error = CustomException(e,sys).error_message
        return handle_my_error(error)

if __name__ == '__main__':
     app.run(host = '0.0.0.0', debug = True, port = 8000)