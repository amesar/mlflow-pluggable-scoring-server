from mlflow_pluggable_scoring_server.plugin import BasePlugin

class Plugin(BasePlugin):

    def load_model(self, model_uri):
        import mlflow.keras
        return mlflow.keras.load_model(model_uri)

    def predict(self, model, img_bytes):
        import json
        import numpy as np
        from io import BytesIO
        from PIL import Image
        img = Image.open(BytesIO(img_bytes))
        img_np = np.resize(img, (1, 28*28))
        data = img_np.astype("float32") / 255
        predictions = model.predict(data)
        return json.dumps(predictions.tolist())

    def request_content_type(self):
        return "application/octet-stream"
