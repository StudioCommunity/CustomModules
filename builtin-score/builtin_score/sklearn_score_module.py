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
        model_file_path = os.path.join(model_path, sklearn_conf[constants.MODEL_FILE_PATH_KEY])
        with open(model_file_path, "rb") as fp:
            self.model = pickle.load(fp)

    
    def run(self, df):
        y = self.model.predict(df)
        return y
