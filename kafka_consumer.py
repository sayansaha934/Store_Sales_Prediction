from pyspark.sql import SparkSession
from pyspark.sql.functions import from_csv, col
from predictFromModel.predictFromModelBulk import prediction

from pyspark.sql.functions import from_csv, col, trim, regexp_replace

schema = "Item_Identifier STRING,Item_Weight FLOAT,Item_Fat_Content STRING,Item_Visibility FLOAT,Item_Type STRING,Item_MRP FLOAT,Outlet_Identifier STRING,Outlet_Establishment_Year INT,Outlet_Size STRING,Outlet_Location_Type STRING,Outlet_Type STRING"
spark = (SparkSession.builder.appName("spark_consumer") \
         .config("spark.jars.packages","org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1")\
         .getOrCreate())

data = (
    spark.readStream.format("kafka")
    .option("kafka.bootstrap.servers", "kafka:9092")
    .option("subscribe", "store_sales_prediction")
    .option("startingOffsets", "earliest")
    .load()
)
data = data.selectExpr("CAST(value AS STRING)")
data = data.select(from_csv(col("value"), schema).alias("records"))


data = data.select("records.*")

data = data.filter("Item_Identifier != 'Item_Identifier'")

def process_batch(batch_df, batch_id):
    batch_df.select('Outlet_Type').show()
    pred = prediction(streaming=True, spark_df=batch_df)
    pred.predictionFromModel()
query = data.writeStream.foreachBatch(process_batch).start()

# Wait for the streaming query to finish
query.awaitTermination()
spark.stop()