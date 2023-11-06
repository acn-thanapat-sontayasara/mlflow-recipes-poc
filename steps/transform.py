from typing import Any
from pyspark.sql import DataFrame
from pyspark.sql.functions import col
from pyspark.sql.functions import avg
from pyspark.sql.functions import when
from pyspark.sql.functions import sum
from pyspark.sql.functions import regexp_extract

# # import: pyspark
# from pyspark.ml.feature import StringIndexer
# from pyspark.ml.feature import VectorAssembler
# from pyspark.ml import Pipeline
# from pyspark.sql import SparkSession

# def transformer_fn():
#     string_indexer = StringIndexer(
#             inputCols=["Sex", "Embarked"],
#             outputCols=["Gender", "Boarded"],
#             handleInvalid="keep",
#         )
#     required_features = ["Pclass", "Age", "Fare", "Gender", "Boarded"]
#     assembler = VectorAssembler(inputCols=required_features, outputCol="features")

#     return Pipeline(stages=[string_indexer, assembler])

# ---------- Pandas version --------------

from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from pandas import DataFrame

def add_outputCol(df: DataFrame):
    df['Gender'] = df['Sex']
    df['Boarded'] = df['Embarked']
    # df['Survived'] = df['Survived'].astype(int)
    df = df.drop(columns=['Sex', 'Embarked'])
    return df

def transformer_fn():
    # Define the stages of the pipeline

    return Pipeline(
        steps=[(
                "add_outputCol",
                FunctionTransformer(add_outputCol),
            ),
            (
                "encoder",
                ColumnTransformer(
                    transformers=[
                        (
                            "ordinal_enc",
                            OrdinalEncoder(),
                            ["Gender", "Boarded"],
                        ),
                    ]
                )
            )
        ]
    )

# from pandas import DataFrame
# import pandas as pd

# def transform_cast_datatype(df: DataFrame):
#     # As noted above, all the columns are of string type.
#     # Some of them need to be cast as numeric data for analysis,
#     # and the necessary conversions are performed.
    # df = df.toPandas()
#     df['Survived'] = df['Survived'].astype(float)
#     df['Pclass'] = df['Pclass'].astype(float)
#     df['Age'] = df['Age'].astype(float)
#     df['Fare'] = df['Fare'].astype(float)

#     return df

# def transform_null_handling(df: DataFrame, replace_with: Any = None):
#     # Eliminate null values
#     return df.replace("?", replace_with).dropna(how="any")

# def transformer_fn(df: DataFrame):
#     df = transform_cast_datatype(df)
#     df = transform_null_handling(df)
#     return df
