# mlflow-pluggable-scoring-server

## Overview

MLfow provides a versatile scoring server based on the MLflow [Pyfunc](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow-pyfunc) flavor. 
The MLflow scoring server accepts only JSON or CSV requests, and JSON responses. 
See [Deploy MLflow models](https://mlflow.org/docs/latest/models.html#deploy-mlflow-models).

This pluggable scoring server is an exploratory POC that offers the ability to plug in custom request or response payloads.
It addresses the need to submit an image for scoring.

## Limitations

* Ideally, once this POC pluggable logic is finalized, it could/should be merged into the MLflow code base.
* The installation of packages is rudimentary since they are installed in the current virtual environment. 
The correct solution is to create a virtual environment on the fly like the MLflow scoring server does.

## Sample plugins

A number of sample plugins are provided for convenience.
Each sample has a plugin.py file, sample data and a sample MLflow model.

Following examples are provided:

|Algorithm |  Plugin | Model | Data | Payloads |
|-----|----------|---------|--|--|
| Keras MNIST |  [plugin.py](plugin_samples/keras_mnist/plugin.py) | [model](plugin_samples/keras_mnist/94580121e06f483691151c8337f64b48/artifacts/keras-model) |  [mnist_0_10.png](plugin_samples/keras_mnist/data/mnist_0_10.png) | PNG image request and JSON response |
| Sklearn Wine | [plugin.py](plugin_samples/sklearn_wine/plugin.py) |  [model](plugin_samples/sklearn_wine/7a7022b7d5ce48e4ac789808c6d3250e/artifacts/sklearn-model) | [predict-wine-quality.json](plugin_samples/data/predict-wine-quality.json) | JSON request and JSON response |
| Sklearn Wine (ONNX)| [plugin.py](plugin_samples/sklearn_wine/onnx_plugin.py) |  [model](plugin_samples/sklearn_wine/7a7022b7d5ce48e4ac789808c6d3250e/artifacts/onnx-model) | [predict-wine-quality.json](plugin_samples/data/predict-wine-quality.json) | JSON request and JSON response |
| Spark ML | [plugin.py](plugin_samples/sparkml_wine/plugin.py) |  N/A | [predict-wine-quality.json](plugin_samples/data/predict-wine-quality.json) | JSON request and JSON response |

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

## Run 

There Two ways to run the scoring server:
  * Local server
  * Docker server

Setup:
```
conda env create --file conda.yaml
source activate mlflow-pluggable-scoring-server
```

## Run Scoring Server Locally

### Usage
```
python -u mlflow_pluggable_scoring_server.webserver --help

Options:
  --host TEXT               Host.
  --port INTEGER            Port.
  --plugin-file TEXT        Plugin path.  [required]
  --plugin-class TEXT       Plugin full class name. Default is 'plugin.Plugin'.
  --model-uri TEXT          Model URI.  [required]
  --packages TEXT           PyPI packages (comma delimited).
  --conf TEXT               Webserver configuration file.
  --help                    Show this message and exit.
```

### Keras MNIST

Run with model from sample run.
```
python -u -m mlflow_pluggable_scoring_server.webserver \
  --host localhost --port 5005 \
  --plugin-file plugin_samples/keras_mnist/plugin.py \
  --model-uri file:plugin_samples/keras_mnist/94580121e06f483691151c8337f64b48/artifacts/keras-model \
  --packages tensorflow==2.3.0,Pillow
```

Run with model from Model Registry.
```
python -u -m mlflow_pluggable_scoring_server.webserver \
  --host localhost --port 5005 \
  --plugin-file plugin_samples/keras_mnist/plugin.py \
  --model-uri models:/keras_mnist/Production \
  --packages tensorflow==2.3.0,Pillow
```

### Sklearn Wine
```
python -u -m mlflow_pluggable_scoring_server.webserver \
  --host localhost --port 5005 \
  --plugin-file plugin_samples/sklearn_wine/plugin.py \
  --model-uri plugin_samples/sklearn_wine/7a7022b7d5ce48e4ac789808c6d3250e/artifacts/sklearn-model \
  --packages scikit-learn==0.20.2 
```

### Sklearn Wine ONNX
```
python -u -m mlflow_pluggable_scoring_server.webserver \
  --host localhost --port 5005 \
  --plugin-file plugin_samples/sklearn_wine/onnx_plugin.py \
  --model-uri plugin_samples/sklearn_wine/7a7022b7d5ce48e4ac789808c6d3250e/artifacts/onnx-model \
  --packages onnx==1.7.0,onnxmltools==1.7.0,onnxruntime==1.4.0
```

### SparkML Wine
```
python -u -m mlflow_pluggable_scoring_server.webserver \
  --host localhost --port 5005 \
  --plugin-file plugin_samples/sparkml_wine/plugin.py \
  --model-uri models:/sparkml_wine/1 \
  --packages pyspark==2.4.5
```


## Run Scoring Server with Docker

### Build Docker Image

See [docker_build.py](mlflow_pluggable_scoring_server/docker_build.py).

#### Usage
```
python -u mlflow_pluggable_scoring_server.docker_build.py --help

Options:
  --packages TEXT           PyPI packages (comma delimited).
  --model-uri TEXT          Model URI.  [required]
  --plugin-file TEXT        Python plugin file.  [required]
  --docker-image TEXT       Docker image name  [required]
  --plugin-full-class TEXT  Plugin full class name. Default is 'plugin.Plugin'.
  --tmp-model-dir TEXT      Temporary model directory for docker COPY. Default is './tmp'.
```

#### Run against sample file-based URI
```
python -u mlflow_pluggable_scoring_server.docker_build.py \
  --plugin-file plugin_samples/keras_mnist/plugin.py \
  --model-uri file:plugin_samples/keras_mnist/94580121e06f483691151c8337f64b48/artifacts/keras-model \
  --packages tensorflow==2.3.0,Pillow \
  --docker-image mlflow_scoring_server_keras
```

#### Run against registered model URI
```
python -u mlflow_pluggable_scoring_server.docker_build.py \
  --plugin-file plugin_samples/keras_mnist/plugin.py \
  --model-uri models:keras_mnist/1 \
  --packages tensorflow==2.3.0,Pillow \
  --docker-image mlflow_scoring_server_keras
```

### Run Docker Container

```
docker container run -p 5005:5005 -dit --name mlflow_scoring_server_keras my_scoring_server_keras
```


## Score with REST API

Score request data.

### Keras MNIST
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


### Wine Quality - Sklearn or Spark ML
```
curl -X POST \
  -H "Content-Type:application/json" \
  -H "Accept:application/json" \
  --data-binary @plugin_samples/data/predict-wine-quality.json \
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
       "/api/status": {
. . .
```
