# mlflow-pluggable-scoring-server

## Overview

MLfow provides a versatile scoring server based on the MLflow pyfunc flavor. In order to support multiple flavors, the MLflow scoring server accepts only JSON or CSV requests, and JSON responses.  See [Deploy MLflow models](https://mlflow.org/docs/latest/models.html#deploy-mlflow-models).

This pluggable scoring server is an exploratory POC that provides the ability to plug in custom request or response payloads.
It addresses the need to submit an image (PNG) for scoring.

## Sample plugins

Following examples are provided:
* Sklearn Wine + ONNX version - JSON request and JSON response.
* Keras MNIST - PNG image request and JSON response.

**Details**

|Algorithm | Config | Plugin | Data |
|-----|----------|---------|---|
| Sklearn Wine | [conf.yaml](samples/sklearn_wine/conf.yaml) | [plugin.py](samples/sklearn_wine/plugin.py) | [predict-wine-quality.json](samples/sklearn_wine/data/predict-wine-quality.json) |
| Sklearn Wine (ONNX)| [conf.yaml](samples/sklearn_wine/onnx/conf.yaml) | [plugin.py](samples/sklearn_wine/onnx/plugin.py) | [predict-wine-quality.json](samples/sklearn_wine/data/predict-wine-quality.json) |
| Keras MNIST | [conf.yaml](samples/keras_mnist/conf.yaml) | [plugin.py](samples/keras_mnist/plugin.py) | [mnist_0_10.png](samples/keras_mnist/data/mnist_0_10.png) |

## Configure

Sample Sklearn [conf.yaml](samples/sklearn_wine/conf.yaml).
```
port: 5000
host: localhost

plugin: samples/sklearn_wine/plugin.py
model_uri: file:samples/sklearn_wine/7a7022b7d5ce48e4ac789808c6d3250e/artifacts/sklearn-model
#model_uri: models:/sklearn_wine/1

packages: [ scikit-learn==0.20.2 ]

logging: {
  level: DEBUG,
  format: '[%(asctime)s] %(levelname)s %(module)s:%(funcName)s:%(lineno)d: %(message)s'
}
```

## Run Scoring Server

**Sklearn**
```
python -u -m mlflow_server.webserver --conf samples/sklearn_win/conf.yaml
```

**Sklearn ONNX**
```
python -u -m mlflow_server.webserver --conf samples/sklearn_win/onnx/conf.yaml
```

**Keras MNIST**
```
python -u -m mlflow_server.webserver --conf samples/keras_mnist/conf.yaml
```

### Options
```
optional arguments:
  -h, --help   show this help message and exit
  --conf CONF  conf file
```


## REST API

### Score

**Sklearn wine example**
```
curl -X POST \
  -H "Content-Type:application/json" \
  -H "Accept:application/json" \
  --data-binary @samples/sklearn_wine/data/predict-wine-quality.json \
  http://localhost:5005/api/predict
```
```
[5.370157819225251, 5.535714285714286, 5.760869565217392]
```

**Keras MNIST example**
```
curl -X POST \
  -H "Content-Type:application/json" \
  -H "Accept:application/octet-stream" \
  --data-binary @samples/keras_mnist/data/mnist_0_10.png \
  http://localhost:5005/api/predict
```
```
[[0.999970555305481, 2.3606221422056706e-09, 7.467354862455977e-06, 1.942700578183576e-07, 2.5061572261897425e-10, 1.1649563930404838e-05, 7.735456165391952e-06, 1.382385335091385e-06, 1.0034289488203285e-07, 8.520780738763278e-07]]
```

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
