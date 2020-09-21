from mlflow_pluggable_scoring_server.plugin import BasePlugin

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("App").getOrCreate()

class Plugin(BasePlugin):

    def load_model(self, model_uri):
        import mlflow.spark
        return mlflow.spark.load_model(model_uri)

    def predict(self, model, data):
        import json
        dct = json.loads(data)
        data = [[float(c) for c in row] for row in dct["data"]] # make sure all ints are floats
        df = spark.createDataFrame(data, dct["columns"])
        predictions = model.transform(df).select("prediction")
        predictions = [ row.asDict()["prediction"] for row in predictions.collect() ]
        return json.dumps(predictions)

    def request_content_type(self):
        return "application/json"
