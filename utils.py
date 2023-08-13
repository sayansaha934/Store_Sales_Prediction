
from pyspark.sql.functions import monotonically_increasing_id

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
