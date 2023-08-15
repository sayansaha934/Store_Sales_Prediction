import os
import pandas as pd
import pickle
import shutil
from data_preprocessing import preprocessing
from model_loading.load_model import Model_Loader
from application_logging.logger import App_Logger
from pyspark.ml.clustering import KMeans,KMeansModel
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from model_path import PreprocessingModelPath
from utils import concat_dataframe
class prediction:

    def __init__(self, path=None, streaming=False, spark_df=None):
        self.path = path
        self.streaming = streaming
        self.spark_df = spark_df
        self.logging = App_Logger()

    def predictionFromModel(self):
        try:
            logging_db = 'Bulk_Data_Logging'
            logging_collection = 'Prediction_Logs'

            self.logging.log(logging_db, logging_collection, 'INFO', 'Prediction Started!!')
            spark = SparkSession.builder.appName("store_sales_prediction").getOrCreate()

            if self.streaming and self.spark_df:
                data = self.spark_df
            else:
                data = spark.read.csv(self.path, header=True, inferSchema=True)

            item_outlet = data.select('Item_Identifier', 'Outlet_Identifier')

            # Data Preprocessing
            self.logging.log(logging_db, logging_collection, 'INFO', 'Data Preprocessing started!!')
            preprocess = preprocessing.Preprocessor(logging_db, logging_collection)
            data = preprocess.dropUnnecessaryColumns(data)
            data = preprocess.editDataset(data)
            
            data = preprocess.indexCategoricalValues(data)
            data = preprocess.imputeMissingValues(data)
            data = preprocess.encodeCategoricalValues(data)
            data = preprocess.scaleNumericalValues(data)
            self.logging.log(logging_db, logging_collection, 'INFO', 'Successful End of Data Preprocessing!!')

            # Clustering
            feature_cols = [col for col in data.columns if col != 'Item_Outlet_Sales']
            vector_assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
            data = vector_assembler.transform(data)
            cluster_model = KMeansModel.load(PreprocessingModelPath.CLUSTER_MODEL.value)
            data = cluster_model.transform(data)

            data = data.withColumnRenamed('prediction', 'Cluster')
            data = data.drop(*feature_cols)
            clusters = sorted(data.select("Cluster").distinct().rdd.flatMap(lambda x: x).collect())

            # Prediction From Model
            self.logging.log(logging_db, logging_collection, 'INFO', 'Prediction from model started!!')

            data = concat_dataframe(data, item_outlet)
            final_output = pd.DataFrame()
            for i in clusters:
                cluster_data = data.filter(data['Cluster'] == i)

                item_outlet = cluster_data.select('Item_Identifier', 'Outlet_Identifier')
                # item_outlet = item_outlet.reset_index(drop=True)
                cluster_data = cluster_data.drop(*['Cluster', 'Item_Identifier', 'Outlet_Identifier'])

                loader = Model_Loader(logging_db, logging_collection)
                model = loader.get_best_model(i)
                output = model.transform(cluster_data)
                output = output.select('prediction')
                output = output.withColumnRenamed('prediction', 'Item_Outlet_Sales')
                output = concat_dataframe(item_outlet, output)
                output = output.toPandas()
                final_output = pd.concat([final_output, output], axis=0, ignore_index=True)
            spark.stop()

            self.logging.log(logging_db, logging_collection, 'INFO', 'Successful End of Prediction from Model!!')

            # Preparation of Folder to send
            self.logging.log(logging_db, logging_collection, 'INFO', 'Started preparation of folder to send!!')
            output_folder = 'Final_Output_Folder'
            if not os.path.isdir(output_folder):
                os.mkdir(output_folder)

            prediction_path = 'Final_Output_Folder/Prediction'
            if not os.path.isdir(prediction_path):
                os.mkdir(prediction_path)

            final_output.to_csv(prediction_path + "/" + "Sales_Prediction.csv", header=True, index=None)
            print(final_output)

            if os.path.isdir("PredictionArchivedBadData"):
                shutil.move('PredictionArchivedBadData', output_folder)

            shutil.make_archive(output_folder, 'zip', output_folder)
            shutil.rmtree(output_folder)
            if self.path:
                os.remove(self.path)

            if os.path.isdir("Prediction_Batch_Files"):
                shutil.rmtree('Prediction_Batch_Files')

            self.logging.log(logging_db, logging_collection, 'INFO', 'Successful End of Preparation of folder to send!!')
            self.logging.log(logging_db, logging_collection, 'INFO', 'Successful End of Prediction')

            spark.stop()
            return output_folder + '.zip'
        except Exception as e:
            spark.stop()
            raise e
