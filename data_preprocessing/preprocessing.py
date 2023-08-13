import pandas as pd
import pickle
from sklearn.impute import KNNImputer
import numpy as np
from application_logging.logger import App_Logger

from utils import modify_columns, concat_dataframe
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType
from pyspark.ml import PipelineModel
from pyspark.ml.feature import  OneHotEncoderModel
from pyspark.ml.feature import StandardScaler, StandardScalerModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import Imputer, ImputerModel
from model_path import PreprocessingModelPath
class Preprocessor:
    '''
    This class will be used to preprocess the data before prediction
    '''

    def __init__(self, logging_db, logging_collection):
        self.logging_db = logging_db
        self.logging_collection = logging_collection
        self.logging = App_Logger()



    def dropUnnecessaryColumns(self, data):
        '''
                    Method Name: dropUnnecessaryColumns
                    Description: It drops Item_Type column
                    Output: A Dataframe without  Item_Type column
                    On Failure: Raise Exception

                    Written by: Sayan Saha
                    Version: 1.0
                    Revision: None
                '''
        try:
            self.logging.log(self.logging_db, self.logging_collection, 'INFO', 'Started to drop  Unnecessary Columns')
            data = data.drop('Item_Type')
            self.logging.log(self.logging_db, self.logging_collection, 'INFO', 'Dropped Unnecessary Columns Successfully!!')
            return data

        except Exception as e:
            self.logging.log(self.logging_db, self.logging_collection, 'ERROR', f"Error occured to drop columns: {e}")
            raise e



    def editDataset(self, data):

        '''
                            Method Name: editDataset
                            Description: Editing Item_Visibility, Item_Type, Item_Fat_Content, Item_Identifier, Outlet_Age columns
                            Output: A Dataframe
                            On Failure: Raise Exception

                            Written by: Sayan Saha
                            Version: 1.0
                            Revision: None
                        '''
        try:
            self.logging.log(self.logging_db, self.logging_collection, 'INFO', 'Started to edit dataset')

            ### Editing dataset
            data = data.withColumn('Item_Visibility', F.when(data['Item_Visibility'] == 0, None).otherwise(data['Item_Visibility']))

            data = data.withColumn('Item_Fat_Content', F.when(
                (data['Item_Fat_Content'] == 'low fat') | (data['Item_Fat_Content'] == 'LF'), 'Low Fat'
            ).otherwise(F.when(data['Item_Fat_Content'] == 'reg', 'Regular').otherwise(data['Item_Fat_Content'])))

            # Apply lambda function on 'Item_Identifier'
            item_identifier_udf = F.udf(lambda x: x[:2], F.StringType())
            data = data.withColumn('Item_Identifier', item_identifier_udf(data['Item_Identifier']))

            # Calculate Outlet_Age and drop Outlet_Establishment_Year
            data = data.withColumn('Outlet_Age', (F.lit(2013) - data['Outlet_Establishment_Year']).cast(IntegerType()))
            data = data.drop('Outlet_Establishment_Year')

            # Update 'Item_Fat_Content' for Item_Identifier=="NC"
            data = data.withColumn('Item_Fat_Content', F.when(data['Item_Identifier'] == "NC", 'Non Edible').otherwise(data['Item_Fat_Content']))
            
            #update  Outlet_Size
            data = data.withColumn('Outlet_Size', F.when(data['Outlet_Size'] == 'Small', 0)
                .when(data['Outlet_Size'] == 'Medium', 1)
                .when(data['Outlet_Size'] == 'High', 2)
                .otherwise(None))

            self.logging.log(self.logging_db, self.logging_collection, 'INFO', 'Dataset edited Successfully!!')
            return data

        except Exception as e:
            self.logging.log(self.logging_db, self.logging_collection, 'ERROR', f"Error occured to edit dataset: {e}")
            raise e


    def indexCategoricalValues(self, data):
        '''
                            Method Name: encodeCategoricalValuesClassification
                            Description: It encodes categorical values
                            Output: A Dataframe with encoded categorical values
                            On Failure: Raise Exception

                            Written by: Sayan Saha
                            Version: 1.0
                            Revision: None
                        '''

        try:
            self.logging.log(self.logging_db, self.logging_collection, 'INFO', 'Started to encode Categorical Values')

            all_columns=data.columns
            # Identify the categorical and numeric columns
            categorical_cols = ['Item_Identifier', 'Item_Fat_Content', 'Outlet_Identifier', 'Outlet_Type', 'Outlet_Location_Type']

            ### Index the categorical columns
            indexer_model = PipelineModel.load(PreprocessingModelPath.INDEXER_MODEL.value)
            data = indexer_model.transform(data)
            data = modify_columns(data, categorical_cols, '_index')

            self.logging.log(self.logging_db, self.logging_collection, 'INFO', 'Encoded Categorical Values Successfully!!')

            return data

        except Exception as e:
            self.logging.log(self.logging_db, self.logging_collection, 'ERROR', f"Error occured to encode Categorical Values: {e}")

            raise e
    
    def encodeCategoricalValues(self, data):
        '''
                            Method Name: encodeCategoricalValuesClassification
                            Description: It encodes categorical values
                            Output: A Dataframe with encoded categorical values
                            On Failure: Raise Exception

                            Written by: Sayan Saha
                            Version: 1.0
                            Revision: None
                        '''

        try:
            self.logging.log(self.logging_db, self.logging_collection, 'INFO', 'Started to encode Categorical Values')
            categorical_cols = ['Item_Identifier', 'Item_Fat_Content', 'Outlet_Identifier', 'Outlet_Type', 'Outlet_Location_Type']
            encoder_model = OneHotEncoderModel.load(PreprocessingModelPath.ENCODER_MODEL.value)
            data = encoder_model.transform(data)
            data = modify_columns(data, categorical_cols, '_encoded')


            self.logging.log(self.logging_db, self.logging_collection, 'INFO', 'Encoded Categorical Values Successfully!!')

            return data

        except Exception as e:
            self.logging.log(self.logging_db, self.logging_collection, 'ERROR', f"Error occured to encode Categorical Values: {e}")

            raise e




    def imputeMissingValues(self, data):
        '''
                                    Method Name: imputeMissingValues
                                    Description: It imputes missing values
                                    Output: A Dataframe with imputed misssing values
                                    On Failure: Raise Exception

                                    Written by: Sayan Saha
                                    Version: 1.0
                                    Revision: None
                                '''

        try:
            self.logging.log(self.logging_db, self.logging_collection, 'INFO', 'Started to impute Missing Values')

            imputed_cols = data.columns
            imputer_model = ImputerModel.load(PreprocessingModelPath.IMPUTER_MODEL.value)
            data = imputer_model.transform(data)
            data = modify_columns(data, imputed_cols, '_imputed')

            # data['Outlet_Size'] = np.round(data['Outlet_Size'])

            self.logging.log(self.logging_db, self.logging_collection, 'INFO', 'Imputed Missing Values Successfully!!')
            return data

        except Exception as e:
            self.logging.log(self.logging_db, self.logging_collection, 'ERROR', f"Error occured to impute Missing Values: {e}")

            raise e



    def scaleNumericalValues(self, data):
        '''
                            Method Name: scaleNumericalValuesClassification
                            Description: It scales numerical values
                            Output: A Dataframe with scaled numerical values
                            On Failure: Raise Exception

                            Written by: Sayan Saha
                            Version: 1.0
                            Revision: None
                '''

        try:
            self.logging.log(self.logging_db, self.logging_collection, 'INFO', 'Started to encode Numerical Values')

            scaling_cols=['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Age']
            assembler = VectorAssembler(inputCols=scaling_cols, outputCol="features")
            data = assembler.transform(data)
            data = data.drop(*scaling_cols)
            scaler_model = StandardScalerModel.load(PreprocessingModelPath.SCALER_MODEL.value)

            # Transform the numerical columns using the scaler
            data = scaler_model.transform(data)
            data = data.drop('features')
            self.logging.log(self.logging_db, self.logging_collection, 'INFO', 'Scaled Numerical Values Successfully!!')
            return data

        except Exception as e:
            self.logging.log(self.logging_db, self.logging_collection, 'ERROR', f"Error occured to Scale Numerical Values: {e}")
            raise e


