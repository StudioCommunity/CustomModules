import torch
from torch.autograd import Variable
import cloudpickle
import importlib
import numpy as np
import pandas as pd
import sys
import pickle
import os

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class PytorchScoreModule(object):
    def __init__(self, model_path, config):
        pt_config = config['pytorch']
        self.wrapper = self.load_from_cloudpickle(model_path, pt_config)

    def run(self, df):
        return self.wrapper.predict(df)
    
    def load_from_cloudpickle(self, model_path, pt_config):
        print('device is gpu ? ', self.is_gpu())
        if self.is_gpu():
            model_file = pt_config.get('cuda_model_file_path', '') # try to get gpu model firstly
            self.is_gpu_model = True

        if model_file != '':
            model_path = os.path.join(model_path, model_file)
        else:
            self.is_gpu_model = False
            model_path = os.path.join(model_path, pt_config['model_file_path'])

        print(f'Load model: is_gpu_model = {self.is_gpu_model}')
        with open(model_path, 'rb') as fp:
            model = cloudpickle.load(fp)
        return PytorchWrapper(model, self.is_gpu_model)

    def is_gpu(self):
        return torch.cuda.is_available()

class PytorchWrapper(object):
    def __init__(self, model, is_gpu_model):
        self.model = model
        self.model.eval()
        self.device = 'cuda' if is_gpu_model else 'cpu'
    
    def predict(self, df):
        print(f"model_is_gpu = {self.device}")
        output = []
        with torch.no_grad():
            print(f"predict df = \n {df}")
            for _, row in df.iterrows():
                input_params = []
                print(f"ROW = \n {row}")
                if self.is_image(row):
                    input_params.append(torch.Tensor(row).to(self.device))
                    print(f"IMAGES: {input_params}")
                else:
                    for entry in row:
                        input_params.append(torch.Tensor(entry).to(self.device))
                    print(f"FEATURES: {input_params}")
                predicted = self.model(*input_params)
                output.append(predicted.cpu().numpy()) # here to cpu, as "can't convert CUDA tensor to numpy"
        output_df = pd.DataFrame(output)
        print(f"output: {output_df}")
        return output_df
    
    def is_image(self, row):
        # TODO:
        return len(row) > 10
