"""
MLflow pluggable scoring server.
"""

import sys
import os
import json
import logging
import click
import importlib
from flask import Flask, request, make_response
from flask_restplus import Api, Resource

app = Flask(__name__)
api = Api(app, version="1.0", title="MLflow Pluggable Scoring API", description="Scoring API")

plugin = None
model = None

# Class name of plugin is fixed.
plugin_full_class= "plugin.Plugin"

def load_class(path, full_class_str):
    if not os.path.exists(path):
        raise Exception(f"File '{path}' does not exist")
    sys.path.append(os.path.dirname(path))
    class_str = full_class_str.split(".")[-1]
    spec = importlib.util.spec_from_file_location(path, path)
    module = spec.loader.load_module(spec.name)
    return getattr(module, class_str)

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
        logging.debug(f"plugin.request_content_type: {plugin.request_content_type()}")
        if (plugin.request_content_type() != content_type):
            return make_error_response(400,f"Request content-type must be '{plugin.request_content_type()}' but found '{content_type}'")

        data = request.get_data()
        logging.debug(f"request.data.type: {type(data)}")
        logging.debug(f"request.data.len: {len(data)}")

        logging.debug(f"plugin: {plugin}")
        rsp = plugin.predict(model, data)
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
@click.option("--conf", help="Config file.", default="conf.yaml", type=str)
def main(conf):
    global plugin, model
    print("Options:")
    for k,v in locals().items():
        print(f"  {k}: {v}")
    import yaml

    # Open config file
    with open(conf, "r") as f:
        conf = yaml.safe_load(f)
    print("conf:",conf)

    # Set logging config
    fmt = "[%(asctime)s] %(levelname)s @ %(module)s:%(funcName)s:%(lineno)d: %(message)s"
    logging.basicConfig(level=conf["logging"]["level"], format=conf["logging"]["format"])
    print("logging_level:",logging.getLevelName(logging.getLogger().getEffectiveLevel()))

    # Install Python packages
    packages = conf["packages"]
    for pkg in packages:
        run_command(f"pip install {pkg}")
 
    # Load model plugin
    model_uri = conf["model_uri"]
    print("model_uri:",model_uri)

    plugin_class = load_class(conf["plugin"], plugin_full_class)
    plugin = plugin_class()
    print("plugin:",plugin)

    model = plugin.load_model(model_uri)
    #print("model:",model)
    print("model.type:",type(model))

    # Run app
    app.run(debug=True, host=conf["host"], port=conf["port"])

if __name__ == "__main__":
    main()
