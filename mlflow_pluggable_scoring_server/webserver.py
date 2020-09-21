"""
MLflow pluggable scoring server.
"""

import sys
import os
import json
import logging
import click
from flask import Flask, request, make_response
from flask_restplus import Api, Resource
from mlflow_pluggable_scoring_server.plugin import load_plugin

app = Flask(__name__)
api = Api(app, version="1.0", title="MLflow Pluggable Scoring API", description="Scoring API")

_plugin = None
_model = None

def run_command(cmd):
    from subprocess import Popen, PIPE
    print("Running command:",cmd)
    proc = Popen(cmd.split(" "), stdout=PIPE, stderr=PIPE, universal_newlines=True)
    proc.wait()
    if (proc.stderr):
        output = proc.stderr.read()
        if len(output) > 0:
            raise Exception(f"Failed to execute command '{cmd}'. Error: {output}")

def make_error_msg(msg):
    return json.dumps({"ERROR": str(msg)})+"\n"

def make_error_response(code, msg):
    logging.error("ERROR: code={} error={}".format(code,msg))
    rsp = make_response(make_error_msg(msg))
    rsp.status_code = code
    return rsp

@api.route("/api/predict")
class Predict(Resource):
    @api.doc(description="Endpoint for prediction.")
    @api.doc(params={"text": "Predict payload and return predictions."})
    def post(self):
        accept = request.headers.get("Accept")
        logging.debug(f"accept: {accept}")
        content_type = request.headers.get("Content-type")
        logging.debug(f"content_type: {content_type}")
        logging.debug(f"plugin.request_content_type: {_plugin.request_content_type()}")
        if (_plugin.request_content_type() != content_type):
            return make_error_response(400,f"Request content-type must be '{_plugin.request_content_type()}' but found '{content_type}'")

        data = request.get_data()
        logging.debug(f"request.data.type: {type(data)}")
        logging.debug(f"request.data.len: {len(data)}")

        logging.debug(f"plugin: {_plugin}")
        rsp = _plugin.predict(_model, data)
        logging.debug(f"rsp: {rsp}")
        return make_response(rsp)

@api.route("/api/status")
class StatusCollection(Resource):
    @api.doc(description="Show server status")
    def get(self):
        dct = { 
            "system": {
                "pid": os.getpid(),
                "current_dir": os.getcwd(),
                "python.version": sys.version,
                "request": {
                    "url_root": request.url_root,
                    "remote_addr": request.remote_addr
                },
                "current_logging_level":  logging.getLevelName(logging.getLogger().getEffectiveLevel())
            }
        }
        return dct

@click.command()
@click.option("--host", help="Host.", default="localhost", type=str)
@click.option("--port", help="Port.", default=5005, type=int)
@click.option("--plugin-path", help="Plugin path.", required=True, type=str)
@click.option("--plugin-full-class", help="Plugin full class name. Default is 'plugin.Plugin'.", default="plugin.Plugin", type=str)
@click.option("--model-uri", help="Model URI.", required=True, type=str)
@click.option("--packages", help="PyPI packages (comma delimited).", default=None, type=str)
@click.option("--conf", help="Webserver configuration file.", required=False, type=str)

def main(conf, host, port, plugin_path, plugin_full_class, model_uri, packages):
    global _plugin, _model
    print("Options:")
    for k,v in locals().items():
        print(f"  {k}: {v}")
    import yaml
    packages = [] if packages is None else packages.split(",") 

    # Set config options
    if conf and os.path.exists(conf):
        with open(conf, "r") as f:
            conf = yaml.safe_load(f)
        print("conf:",conf)
        log_level = conf["logging"]["level"]
        log_format = conf["logging"]["format"]
        logging.basicConfig(level=conf["logging"]["level"], format=conf["logging"]["format"])
    else:
        log_level = "INFO"
        log_format = "[%(asctime)s] %(levelname)s @ %(module)s:%(funcName)s:%(lineno)d: %(message)s"

    # Set logging config
    logging.basicConfig(level=log_level, format=log_format)
    logging.info(f"Log Level: {log_level}")
    logging.info(f"Log EffectiveLevel: {logging.getLevelName(logging.getLogger().getEffectiveLevel())}")

    # Install Python packages
    for pkg in packages:
        run_command(f"pip install {pkg}")
 
    # Load model plugin
    _plugin = load_plugin(plugin_path, plugin_full_class)
    logging.info(f"plugin.type: {type(_plugin)}")

    _model = _plugin.load_model(model_uri)
    logging.info(f"model.type: {type(_model)}")

    # Run app
    app.run(debug=True, host=host, port=port)

if __name__ == "__main__":
    main()
