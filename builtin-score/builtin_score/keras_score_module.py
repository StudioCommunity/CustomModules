import os

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model

from . import constants


class KerasScoreModule(object):
    
    def __init__(self, model_path, config):
        keras_conf = config["keras"]
        model_file_path = os.path.join(model_path, keras_conf[constants.MODEL_FILE_PATH_KEY])
        self.model = load_model(model_file_path)
        print(f"Successfully loaded model from {model_path}")

    def run(self, df):
        y = self.model.predict(df.values)
        df_output = pd.DataFrame(data=y)
        return df_output
