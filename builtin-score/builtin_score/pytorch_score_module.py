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
        serializer = pt_config.get('serialization_format', 'cloudpickle')
        if serializer == 'cloudpickle':
            self.wrapper = self.load_from_cloudpickle(model_path, pt_config)
        elif serializer == 'savedmodel':
            self.wrapper = self.load_from_savedmodel(model_path, pt_config)
        else:
            self.wrapper = self.load_from_saveddict(model_path, pt_config)

    def run(self, df):
        return self.wrapper.predict(df)
    
    def load_from_cloudpickle(self, model_path, pt_config):
        model_file = pt_config['model_file_path']
        if model_file not in model_path:
            model_path = os.path.join(model_path, model_file)
        with open(model_path, 'rb') as fp:
            model = cloudpickle.load(fp)
        return PytorchWrapper(model, self.is_gpu())

    def load_from_savedmodel(self, model_path, pt_config):
        model_file = pt_config['model_file_path']
        if model_file not in model_path:
            model_path = os.path.join(model_path, model_file)
        model_class_package = pt_config['model_class_package']
        module = importlib.import_module(model_class_package)
        self._set_sys_modules(model_class_package, module)
        model = torch.load(model_path)
        return PytorchWrapper(model, self.is_gpu())
    
    def load_from_saveddict(self, model_path, pt_config):
        model_file = pt_config['model_file_path']
        if model_file in model_path:
            model_path = model_path.replace(model_file, '')
        model_class_package = pt_config['model_class_package']
        model_class_name = pt_config['model_class_name']
        model_class_init_args = os.path.join(model_path, pt_config['model_class_init_args'])
        model_fullpath = os.path.join(model_path, model_file)
        module = importlib.import_module(model_class_package)
        model_class = getattr(module, model_class_name)
        print(f'model_class_init_args = {model_class_init_args}')
        with open(model_class_init_args, 'rb') as fp:
            args = pickle.load(fp)
        model = model_class(*args)
        model.load_state_dict(torch.load(model_fullpath))
        return PytorchWrapper(model, self.is_gpu())
        
    def is_gpu(self):
        return torch.cuda.is_available()

    def _set_sys_modules(self, package, module):
        entries = package.split('.')
        for i in range(len(entries)):
            sys.modules['.'.join(entries[i:])] = module
            print(f"{'.'.join(entries[i:])} : {sys.modules['.'.join(entries[i:])]}")

class PytorchWrapper(object):
    def __init__(self, model, is_gpu):
        self.model = model
        self.is_gpu = is_gpu
        self.device = 'cuda' if is_gpu else 'cpu'
    
    def predict(self, df):
        output = []
        with torch.no_grad():
            logger.info(f"DF = \n {df}")
            print(f"DF1 = \n {df}")
            for _, row in df.iterrows():
                input_params_cpu = []
                input_params_gpu = []
                logger.info(f"ROW = \n {row}")
                print(f"ROW1 = \n {row}")
                if self.is_image(row):
                    input_params_cpu.append(torch.Tensor(row).to('cpu'))
                    input_params_gpu.append(torch.Tensor(row).to('cuda'))
                    print(f"IMAGES: {input_params_cpu}")
                else:
                    for entry in row:
                        input_params_cpu.append(torch.Tensor(entry).to('cpu'))
                        input_params_gpu.append(torch.Tensor(entry).to('cuda'))
                    print(f"FEATURES: {input_params_cpu}")
                try:
                    print("CPU!")
                    predicted = self.model(*input_params_cpu)
                    print("CPU!!")
                except:
                    print("GPU!")
                    predicted = self.model(*input_params_gpu)
                    print("GPU!!")
                output.append(predicted.cpu().numpy())
        return pd.DataFrame(output)
    
    def is_image(self, row):
        # TODO:
        return len(row) > 10
