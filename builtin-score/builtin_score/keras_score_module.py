import os

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from keras import backend as K
from . import constants


class KerasScoreModule(object):
    
    def __init__(self, model_path, config):
        keras_conf = config["keras"]
        serializer = keras_conf.get('serialization_format', 'load_model')
        if serializer == 'load_model':
            self.model = load_model(os.path.join(model_path, keras_conf[constants.MODEL_FILE_PATH_KEY]))
        elif serializer == 'load_weights':
            self.load_model_via_weights(model_path, keras_conf)
        print(f"Successfully loaded model from {model_path}")

    def run(self, df):
        df_output = pd.DataFrame([])
        for _, row in df.iterrows():
            input_params = []
            #print(f"Row = \n {row}")
            if(self.is_image(row)):
                tensor_row = tf.convert_to_tensor(row)
                input_row = K.reshape(tensor_row,(1, -1))
                input_params.append(input_row)
            else:
                for input_arg in row:
                    tensor_arg = tf.convert_to_tensor(input_arg)
                    input_params.append(tensor_arg)
            y_output = self.model.predict(input_params, steps = 1)
            tensor_row_output = y_output.reshape(1, -1)
            df_output = df_output.append(pd.DataFrame(tensor_row_output), ignore_index=True)

        return df_output

    def load_model_via_weights(self, model_path, config):
        model_yaml_file = config.get('model_yaml_file','')
        model_json_file = config.get('model_json_file','')
        model_data = ''
        if model_json_file != '':
            import json
            from keras.models import model_from_json
            with open(os.path.join(model_path, model_json_file), 'r') as f:
                model_data = json.load(f)
            self.model = model_from_json(model_data)
        elif model_yaml_file != '':
            import yaml
            from keras.models import model_from_yaml
            with open(os.path.join(model_path, model_yaml_file), 'r') as f:
                model_data = yaml.safe_load(f)
            self.model = model_from_yaml(model_data)
        else:
            raise Exception(f"Unable to load model, config = {config}")
        
        model_weights_file = config.get('model_weights_file','')
        if model_weights_file == '':
            raise Exception(f"model_weights_file is empty, config =  {config}")

        self.model.load_weights(os.path.join(model_path, model_weights_file))
    
    def is_image(self, row):
        # TO DO:
        if(len(row)>100):
            return True
        else:
            return False
