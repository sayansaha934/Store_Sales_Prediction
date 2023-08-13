import pandas as pd
import numpy as np
import pickle
from data_preprocessing import preprocessing
from model_loading.load_model import Model_Loader
from application_logging.logger import App_Logger

from pyspark.sql import SparkSession
from utils import concat_dataframe
from pyspark.ml.clustering import KMeans,KMeansModel
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import Row
from model_path import PreprocessingModelPath

class prediction:

    def __init__(self, data):
        self.data = data
        self.logging = App_Logger()

    def predictionFromModel(self):
        try:
            spark = SparkSession.builder.appName("single_prediction_session").getOrCreate()
            logging_db='Single_Data_Logging'
            logging_collection='Prediction_Logs'

            self.logging.log(logging_db, logging_collection, 'INFO', 'Prediction Started!!')
            self.logging.log(logging_db, logging_collection, 'INFO', f"user input: {self.data}")
            data = spark.createDataFrame([self.data])


            #Data Preprocessing
            self.logging.log(logging_db, logging_collection, 'INFO', 'Data Preprocessing started!!')
            preprocess = preprocessing.Preprocessor(logging_db, logging_collection)
            data = preprocess.dropUnnecessaryColumns(data)
            data = preprocess.editDataset(data)
            data = preprocess.indexCategoricalValues(data)
            data = preprocess.encodeCategoricalValues(data)
            # data=preprocess.imputeMissingValues(data)
            data = preprocess.scaleNumericalValues(data)
            self.logging.log(logging_db, logging_collection, 'INFO', 'Successful End of Data Preprocessing!!')

            #Clustering
            feature_cols = [col for col in data.columns if col != 'Item_Outlet_Sales']
            vector_assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
            data = vector_assembler.transform(data)
            cluster_model = KMeansModel.load(PreprocessingModelPath.CLUSTER_MODEL.value)
            data = cluster_model.transform(data)

            data = data.withColumnRenamed('prediction', 'Cluster')
            data = data.drop(*feature_cols)
            cluster = data.select('Cluster').rdd.flatMap(lambda x: x).collect()[0]


            #Prediction From Model
            loader = Model_Loader(logging_db, logging_collection)
            model =  loader.get_best_model(cluster)
            prediction = model.transform(data)
            output = prediction.select('prediction').rdd.flatMap(lambda x: x).collect()[0]
            output = np.round(output,4)
            spark.stop()

            self.logging.log(logging_db, logging_collection, 'INFO', f"Prediction output: {output}")
            self.logging.log(logging_db, logging_collection, 'INFO', 'Successful End of Prediction')

            return output

        except Exception as e:
            print(str(e))
            spark.stop()
            raise e




