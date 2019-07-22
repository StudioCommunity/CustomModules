import os

import numpy as np
import pandas as pd
import sklearn
from sklearn.externals import joblib
import pickle

from . import constants


class SklearnScoreModule(object):
    
    def __init__(self, model_path, config):
        sklearn_conf = config["sklearn"]
        model_file_path = os.path.join(model_path, sklearn_conf["model_file_path"])
        DEFAULT_SERIALIZATION_METHOD = "pickle"
        serialization_method = sklearn_conf.get(constants.SERIALIZATION_METHOD_KEY)
        if serialization_method is None:
            print(f"Using default deserializtion method: {DEFAULT_SERIALIZATION_METHOD}")
            serialization_method = pickle
        if serialization_method == "joblib":
            self.model = joblib.load(model_file_path)
        elif serialization_method == "pickle":
            with open(model_file_path, "rb") as fp:
                self.model = pickle.load(fp)
        else:
            raise Exception(f"Unrecognized serializtion format {serialization_method}")
    
    def run(self, df):
        y = self.model.predict(df)
        return y
