# import: standard
import pandas

# import: pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType


def read_spark_df_return_pandas(location: str, file_format: str) -> pandas.DataFrame:
    
    spark = SparkSession.getActiveSession()

    spark_df = (
        spark.table("sample_dataset_db.titanic_feature_engineering")
        .where(col("Survived").isNotNull())
        .withColumn("Survived", col("Survived").cast(IntegerType()))
    )

    return spark_df.toPandas()