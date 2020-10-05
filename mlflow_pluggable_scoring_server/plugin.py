"""
Base plugin class for pluggable scoring.
"""
from abc import abstractmethod, ABCMeta

def load_plugin(path, plugin_class):
    """
    Instantiate a plugin object from a Python file.
    :param path: File path to plugin python file.
    :param plugin_class: Full class path of plugin.
    :return: Plugin object.
    """
    import os
    import sys
    import importlib
    if not os.path.exists(path):
        raise Exception(f"File '{path}' does not exist")
    sys.path.append(os.path.dirname(path))
    class_str = plugin_class.split(".")[-1]
    spec = importlib.util.spec_from_file_location(path, path)
    module = spec.loader.load_module(spec.name)
    plugin = getattr(module, class_str)
    return plugin()


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
