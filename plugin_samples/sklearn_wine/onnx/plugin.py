from mlflow_pluggable_scoring_server.plugin import BasePlugin

class Plugin(BasePlugin):

    def load_model(self, model_uri):
        import mlflow.onnx
        return mlflow.onnx.load_model(model_uri)

    def predict(self, model, data):
        import onnxruntime as rt
        import pandas as pd
        import numpy as np
        import json
        data = data.decode("utf-8")
        data = pd.read_json(data, orient="split")
        if "quality" in data:
             data = data.drop(["quality"], axis=1)
        session = rt.InferenceSession(model.SerializeToString())
        input_name = session.get_inputs()[0].name
        predictions =  session.run(None, {input_name: data.to_numpy().astype(np.float32)})[0]
        return json.dumps(predictions.tolist())

    def request_content_type(self):
        return "application/json"
