# mlflow-pluggable-scoring-server

## Overview

MLfow provides a versatile scoring server based on the MLflow pyfunc flavor. In order to support multiple flavors, the MLflow scoring server accepts only JSON or CSV requests, and JSON responses.  See [Deploy MLflow models](https://mlflow.org/docs/latest/models.html#deploy-mlflow-models).

This pluggable scoring server is an exploratory POC that provides the ability to plug in custom request or response payloads.
It addresses the need to submit an image for scoring.

Ideally, once this POC pluggable logic is finalized, it should ideally be merged into the MLflow code base.

## Sample plugins

Following examples are provided:
* Keras MNIST - PNG image request and JSON response.
* Sklearn Wine + ONNX version - JSON request and JSON response.

Each plugin sample has a plugin.py file, sample data and a sample MLflow run.

**Details**

|Algorithm |  Plugin | Data |
|-----|----------|---------|
| Keras MNIST |  [plugin.py](plugin_samples/keras_mnist/plugin.py) | [mnist_0_10.png](plugin_samples/keras_mnist/data/mnist_0_10.png) |
| Sklearn Wine | [plugin.py](plugin_samples/sklearn_wine/plugin.py) | [predict-wine-quality.json](plugin_samples/sklearn_wine/data/predict-wine-quality.json) |
| Sklearn Wine (ONNX)| [plugin.py](plugin_samples/sklearn_wine/onnx/plugin.py) | ibid |

## Plugin

### Base Plugin Class

The plugin class has the following methods:
* load_model - loads a model specified by the model_uri.
* predict - scores the input, first doing any input data conversion.
* request_content_type - Content type of request.

From [mlflow_pluggable_scoring_server/plugin.py](mlflow_pluggable_scoring_server/plugin.py).

```
"""
Base plugin class for pluggable scoring.
"""
#from collection.abc import abstractmethod, ABCMeta
from abc import abstractmethod, ABCMeta

class BasePlugin(metaclass=ABCMeta):

    @abstractmethod
    def load_model(self, model_uri):
        """
        Load a model.
        :param model_uri: Standard MLflow model URI.
        :return: Model.
        """


    @abstractmethod
    def predict(self, model, data):
        """
        Predict the input data.
        :param model: Model to predict with.
        :param data: Data to predict.
        :return: Predictions.
        """

    @abstractmethod
    def request_content_type(self):
        """
        Content type of request entity body.
        """

    def __repr__(self):
        return f"Request content-type: {self.request_content_type()}"
```

### Example Plugin

From [plugin_samples/keras_mnist/plugin.py](plugin_samples/keras_mnist/plugin.py).
We first convert the input PNG image to a numpy array and then score.

```
from mlflow_pluggable_scoring_server.plugin import BasePlugin
  
class Plugin(BasePlugin):
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

    def load_model(self, model_uri):
        import mlflow.keras
        return mlflow.keras.load_model(model_uri)

    def request_content_type(self):
        return "application/octet-stream"
```

## Run Scoring Server

### Keras MNIST

Run with model from sample run.
```
python -u -m mlflow_pluggable_scoring_server.webserver \
  --host localhost --port 5005 \
  --plugin plugin_samples/keras_mnist/plugin.py \
  --model-uri file:plugin_samples/keras_mnist/94580121e06f483691151c8337f64b48/artifacts/keras-model \
  --packages tensorflow==2.3.0,Pillow
```

Run with model from Model Registry.
```
python -u -m mlflow_pluggable_scoring_server.webserver \
  --host localhost --port 5005 \
  --plugin plugin_samples/keras_mnist/plugin.py \
  --model-uri models:/keras_mnist/Production \
  --packages tensorflow==2.3.0,Pillow
```

### Sklearn
```
python -u -m mlflow_pluggable_scoring_server.webserver \
  --host localhost --port 5005 \
  --plugin plugin_samples/sklearn_wine/plugin.py \
  --model-uri plugin_samples/sklearn_wine/7a7022b7d5ce48e4ac789808c6d3250e/artifacts/sklearn-model \
  --packages scikit-learn==0.20.2 
```

### Sklearn ONNX
```
python -u -m mlflow_pluggable_scoring_server.webserver \
  --host localhost --port 5005 \
  --plugin plugin_samples/sklearn_wine/onnx/plugin.py \
  --model-uri plugin_samples/sklearn_wine/7a7022b7d5ce48e4ac789808c6d3250e/artifacts/onnx-model \
  --packages onnx==1.7.0,onnxmltools==1.7.0,onnxruntime==1.4.0
```

### Options
```
Options:
  --host TEXT       Host.
  --port INTEGER    Port.
  --plugin TEXT     plugin.  [required]
  --model-uri TEXT  Model URI.  [required]
  --packages TEXT   PyPI packages (comma delimited).
  --conf TEXT       Webserver configuration file.
```



## Score with REST API

Score request data.

#### Keras MNIST
```
curl -X POST \
  -H "Content-Type:application/octet-stream" \
  -H "Accept:application/json" \
  --data-binary @plugin_samples/keras_mnist/data/mnist_0_10.png \
  http://localhost:5005/api/predict
```
```
[[0.999970555305481, 2.3606221422056706e-09, 7.467354862455977e-06, 1.942700578183576e-07, 2.5061572261897425e-10, 1.1649563930404838e-05, 7.735456165391952e-06, 1.382385335091385e-06, 1.0034289488203285e-07, 8.520780738763278e-07]]
```


#### Sklearn wine
```
curl -X POST \
  -H "Content-Type:application/json" \
  -H "Accept:application/json" \
  --data-binary @plugin_samples/sklearn_wine/data/predict-wine-quality.json \
  http://localhost:5005/api/predict
```
```
[5.370157819225251, 5.535714285714286, 5.760869565217392]
```

## Other REST API endpoints

### Status

Display system status.

```
curl http://localhost:5005/api/status
```
```
{
    "system": {
        "pid": 68015,
        "current_dir": "/opt/mlflow-pluggable-scoring-server",
        "python.version": "3.7.6 (default, Jan  8 2020, 13:42:34) \n[Clang 4.0.1 (tags/RELEASE_401/final)]",
        "request": {
            "url_root": "http://localhost:5005/",
            "remote_addr": "127.0.0.1"
        },
        "current_logging_level": "DEBUG"
    }
}
```

### Swagger

Show REST API with Swagger.

```
curl http://localhost:5005/api/swagger
```
```
{
    "swagger": "2.0",
    "basePath": "/",
    "paths": {
        "/api/predict": {
            "post": {
                "responses": {
                    "200": {
                        "description": "Success"
                    }
                },
                "description": "Endpoint for scoring.",
                "operationId": "post_predict",
                "parameters": [
                    {
                        "description": "Details",
                        "name": "text",
                        "type": "string",
                        "in": "query"
                    }
                ],
                "tags": [
                    "default"
                ]
            }
        },
```
