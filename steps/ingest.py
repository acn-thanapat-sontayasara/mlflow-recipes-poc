from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType


def read_spark():
    return (
        spark.table("sample_dataset_db.titanic_feature_engineering")
        .withColumn("Survived", col("Survived").cast(IntegerType()))
        .where(col("Survived").isNotNull())
    )