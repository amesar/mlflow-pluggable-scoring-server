from mlflow_pluggable_scoring_server.plugin import BasePlugin

class Plugin(BasePlugin):
    def predict(self, model, data):
        import json
        import pandas as pd
        data = data.decode("utf-8")
        data = pd.read_json(data, orient="split")
        if "quality" in data:
             data = data.drop(["quality"], axis=1)
        predictions = model.predict(data)
        return json.dumps(predictions.tolist())

    def load_model(self, model_uri):
        import mlflow.sklearn
        return mlflow.sklearn.load_model(model_uri)

    def request_content_type(self):
        return "application/json"
