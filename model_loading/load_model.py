import os
import pickle
from application_logging.logger import App_Logger
from pyspark.ml import Pipeline, PipelineModel
from model_path import PredictionModelPath
from pyspark.ml.regression import RandomForestRegressor, RandomForestRegressionModel

class Model_Loader:
    '''
    This class shall be used to load model
    '''

    def __init__(self, logging_db, logging_collection):
        self.logging_db = logging_db
        self.logging_collection = logging_collection
        self.logging = App_Logger()

    def get_best_model(self, cluster):
        '''
        Method Name: get_best_model
        Description: It finds the model based on the given cluster and load that model
        Output: model
        On  Failure: Raise Exception

        Written by: Sayan Saha
        Version: 1.0
        Revision: None
        '''
        try:
            self.logging.log(self.logging_db, self.logging_collection, 'INFO', 'Started to find best model')

            for file in os.listdir(PredictionModelPath.PATH.value):
                modelName=file.split('.')[0]
                if modelName.endswith(str(cluster)):
                    model=PipelineModel.load(f'{PredictionModelPath.PATH.value}/{file}')
                    # model=RandomForestRegressionModel.load(f'{PredictionModelPath.PATH.value}/{file}')
                    break
            self.logging.log(self.logging_db, self.logging_collection, 'INFO', 'Got best model and returned the best model successfully!!')

            return model

        except Exception as e:
            self.logging.log(self.logging_db, self.logging_collection, 'ERROR', f"Error occured to get best model: {e}")

            raise e