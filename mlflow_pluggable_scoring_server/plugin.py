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
