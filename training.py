from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.ml.feature import StringIndexer, OneHotEncoder, OneHotEncoderModel
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql.functions import col, isnan, when, count
from pyspark.ml import Pipeline, PipelineModel
import os
import pickle
import  numpy as np
from pyspark.ml.clustering import KMeans,KMeansModel
from pyspark.ml.feature import Imputer, ImputerModel
from pyspark.ml.regression import RandomForestRegressor

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
from xgboost.spark import SparkXGBRegressor

#creating spark session
spark = SparkSession.builder.appName("store_sales_prediction").getOrCreate()

# Load the data into a Spark DataFrame
train_path = 'Train.csv'
df = spark.read.csv(train_path, header=True, inferSchema=True)

# Split the data into training and testing sets
train_df, test_df = df.randomSplit([0.8, 0.2], seed=100)

# Extract x_train, x_test, y_train, and y_test as needed
x_train = train_df.drop("Item_Outlet_Sales")
y_train = train_df.select('Item_Outlet_Sales')
x_test = test_df.drop("Item_Outlet_Sales")
y_test = test_df.select('Item_Outlet_Sales')

def concat_dataframe(df1, df2):
    # Add index column to x_train
    df1 = df1.withColumn("index", monotonically_increasing_id())
    
    # Add index column to y_train
    df2 = df2.withColumn("index", monotonically_increasing_id())
    
    # Perform an inner join on the index column
    df = df1.join(df2, "index").drop("index")
    return df

def modify_columns(df, col_names, extension):
    df = df.drop(*col_names)
    for col in col_names:
        df = df.withColumnRenamed(col + extension, col)
    return df

def cluster_data(data, train=True):
    # Prepare features using VectorAssembler
    output_df=data.select('Item_Outlet_Sales')
    feature_cols = [col for col in data.columns if col != 'Item_Outlet_Sales']
    vector_assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    data_with_features = vector_assembler.transform(data.drop('Item_Outlet_Sales'))
    if train:
      silhouette_scores = {}
      for i in range(2,3):  # Start from k=2 instead of k=1
          kmeans = KMeans().setK(i).setSeed(1)
          model = kmeans.fit(data_with_features.repartition(i))

          # Compute the Silhouette score
          evaluator = ClusteringEvaluator()
          predictions = model.transform(data_with_features)
          silhouette_score = evaluator.evaluate(predictions)
          silhouette_scores[silhouette_score]=i
      optimal_cluster = silhouette_scores[max(silhouette_scores.keys())]
      cluster_model = KMeans().setK(optimal_cluster).setSeed(1)
      cluster_model = cluster_model.fit(data_with_features.repartition(optimal_cluster))
      cluster_model.write().overwrite().save('cluster_model')
    else:
      cluster_model = KMeansModel.load('cluster_model')
    data_with_clusters = cluster_model.transform(data_with_features)
    
    
    data_with_clusters = data_with_clusters.drop('features')
    data_with_clusters = data_with_clusters.withColumnRenamed('prediction', 'Cluster')
    data = concat_dataframe(data_with_clusters, output_df)

    return data

def preprocessing_pipeline(x_train, y_train, train=True):
    data = concat_dataframe(x_train, y_train)

    
    # Dropping unnecessary column
    data = data.drop('Item_Type')
    
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
    # Encoding categorical values
    data = data.withColumn('Outlet_Size', F.when(data['Outlet_Size'] == 'Small', 0)
                         .when(data['Outlet_Size'] == 'Medium', 1)
                         .when(data['Outlet_Size'] == 'High', 2)
                         .otherwise(None))
    all_columns=data.columns
    # Identify the categorical and numeric columns
    categorical_cols = ['Item_Identifier', 'Item_Fat_Content', 'Outlet_Identifier', 'Outlet_Type', 'Outlet_Location_Type']
    numeric_cols = [col for col in all_columns if col not in categorical_cols and col != 'Item_Outlet_Sales']
    
    ### Index the categorical columns
    if train:
        indexers = [StringIndexer(inputCol=col, outputCol=col+'_index', handleInvalid='skip') for col in categorical_cols]
        indexer_pipeline = Pipeline(stages=indexers)
        indexer_model = indexer_pipeline.fit(data)
        indexer_model.write().overwrite().save('indexer_model')
    else:
        indexer_model = PipelineModel.load('indexer_model')
    data = indexer_model.transform(data)
    data = modify_columns(data, categorical_cols, '_index')
    

    ### Impute missing values in all columns
    imputed_cols = data.columns
    if train:
        imputer = Imputer(inputCols=imputed_cols, outputCols=[col + '_imputed' for col in data.columns], strategy='mean')
        imputer_model = imputer.fit(data)
        imputer_model.write().overwrite().save('imputer_model')
    else:
        imputer_model = ImputerModel.load('imputer_model')
    data = imputer_model.transform(data)
    data = modify_columns(data, imputed_cols, '_imputed')
    
    
    
    ###onehot encodding
    if train:
        encoder = OneHotEncoder(inputCols=categorical_cols, outputCols=[col + '_encoded' for col in categorical_cols], dropLast=False)
        encoder_model = encoder.fit(data)
        encoder_model.write().overwrite().save('encoder_model')
    else:
        encoder_model = OneHotEncoderModel.load('encoder_model')
    data = encoder_model.transform(data)
    data = modify_columns(data, categorical_cols, '_encoded')
    
    if train:
      data = cluster_data(data, train=True)
    else:
      data = cluster_data(data, train=False)
    return data

train_df=preprocessing_pipeline(x_train, y_train, train=True)
test_df=preprocessing_pipeline(x_test, y_test, train=False)
def get_best_param_xgb(train_df, test_df):
    # Remove the target column from the input feature set.
    featuresCols = train_df.columns
    featuresCols.remove('Item_Outlet_Sales')
     
    # vectorAssembler combines all feature columns into a single feature vector column, "rawFeatures".
    vectorAssembler = VectorAssembler(inputCols=featuresCols, outputCol="features")
    xgb_regressor = SparkXGBRegressor(num_workers=2, label_col="Item_Outlet_Sales", missing=0.0)
    
    paramGrid = ParamGridBuilder()\
      .addGrid(xgb_regressor.max_depth, [2, 5])\
      .addGrid(xgb_regressor.n_estimators, [10, 100])\
      .build()
     
    # Define an evaluation metric.  The CrossValidator compares the true labels with predicted values for each combination of parameters, and calculates this value to determine the best model.
    evaluator = RegressionEvaluator(metricName="r2",
                                    labelCol=xgb_regressor.getLabelCol(),
                                    predictionCol=xgb_regressor.getPredictionCol())
     
    # Declare the CrossValidator, which performs the model tuning.
    cv = CrossValidator(estimator=xgb_regressor, evaluator=evaluator, estimatorParamMaps=paramGrid)
    
    pipeline = Pipeline(stages=[vectorAssembler, cv])
    pipelineModel = pipeline.fit(train_df)
    predictions = pipelineModel.transform(test_df)
    r2 = evaluator.evaluate(predictions)

    return pipelineModel, r2

def get_best_param_rf(train_df, test_df):
    # Remove the target column from the input feature set.
    featuresCols = train_df.columns
    featuresCols.remove('Item_Outlet_Sales')
     
    # vectorAssembler combines all feature columns into a single feature vector column, "rawFeatures".
    vectorAssembler = VectorAssembler(inputCols=featuresCols, outputCol="features")
    rf = RandomForestRegressor(labelCol='Item_Outlet_Sales')
    
    paramGrid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [10,20]) \
        .addGrid(rf.maxDepth, [5, 10]) \
        .build()
     
    # Define an evaluation metric.  The CrossValidator compares the true labels with predicted values for each combination of parameters, and calculates this value to determine the best model.
    evaluator = RegressionEvaluator(metricName="r2",
                                    labelCol=rf.getLabelCol(),
                                    predictionCol=rf.getPredictionCol())
     
    # Declare the CrossValidator, which performs the model tuning.
    cv = CrossValidator(estimator=rf, evaluator=evaluator, estimatorParamMaps=paramGrid)
    
    pipeline = Pipeline(stages=[vectorAssembler, cv])
    pipelineModel = pipeline.fit(train_df)
    predictions = pipelineModel.transform(test_df)
    r2 = evaluator.evaluate(predictions)

    return pipelineModel, r2

clusters = sorted(train_df.select("Cluster").distinct().rdd.flatMap(lambda x: x).collect())

def process_cluster(cluster_id, train_df, test_df):
    train = train_df.filter(train_df['Cluster'] == cluster_id)
    test = test_df.filter(test_df['Cluster'] == cluster_id)
    train = train.drop('Cluster')
    test = test.drop('Cluster')

    model_xgb, r2_xgb = get_best_param_xgb(train, test)
    model_rf, r2_rf = get_best_param_rf(train, test)

    if r2_xgb > r2_rf:
        best_model, best_model_name = model_xgb, 'XGB'
    else:
        best_model, best_model_name = model_rf, 'RF'
    
    print(f'Cluster: {cluster_id}  r2_xgb = {r2_xgb}  r2_rf = {r2_rf}')
    
    if not os.path.isdir('Models'):
        os.mkdir('Models')
    best_model_path = f'Models/{best_model_name}{cluster_id}'
    best_model.write().overwrite().save(best_model_path)


for cluster_id in clusters:
    process_cluster(cluster_id, train_df, test_df)