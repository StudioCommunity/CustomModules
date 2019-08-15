import importlib
import imp
import numpy as np
import pandas as pd
import sys
import pickle
import os
import ast
import cloudpickle
import logging
import builtin_models.python

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class PythonWrapper(object):
    def __init__(self, model, input_args):
        self.model = model
        self.input_args = input_args
        print(f"input_args = {self.input_args}")
    
    def predict(self, df):
        # todo check input column has all columns in input_args
        input_params = list(df[self.input_args].transpose().values)
        print(f"FEATURES: {input_params}")
        predicted = self.model.predict(*input_params)

        output_df = pd.DataFrame(predicted)
        print(f"output_df:\n{output_df}")
        return output_df


class PythonScoreModule(object):
    def __init__(self, model_path, config):
        self.model = builtin_models.python.load_model(model_path)
        input_args = config.get('inputs', None)
        if(input_args == None): raise ValueError("inputs must be set in model spec")
        self.wrapper = PythonWrapper(self.model, input_args)

    def run(self, df):
        return self.wrapper.predict(df)

