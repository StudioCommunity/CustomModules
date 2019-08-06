import torch
from torch.autograd import Variable
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
        self.model_file = pt_config['model_file_path']
        serializer = pt_config.get('serialization_format', 'cloudpickle')
        if serializer == 'cloudpickle':
            model = self.load_from_cloudpickle(model_path, pt_config)
        elif serializer == 'savedmodel':
            model = self.load_from_savedmodel(model_path, pt_config)
        elif serializer == 'saveddict':
            model = self.load_from_saveddict(model_path, pt_config)
        else:
            raise Exception(f"Unrecognized serializtion format {serialization_method}")

        print('Load model success.')
        is_gpu = torch.cuda.is_available()   
        if is_gpu:
            model = model.to('cuda')
            print('Device cuda is available, convert model from cpu version to gpu version.')
        self.wrapper = PytorchWrapper(model, is_gpu)


    def run(self, df):
        return self.wrapper.predict(df)


    def load_from_cloudpickle(self, model_path, pt_config):
        model_path = os.path.join(model_path, pt_config['model_file_path'])
        with open(model_path, 'rb') as fp:
            import cloudpickle
            return cloudpickle.load(fp)
         

    def load_from_savedmodel(self, model_path, pt_config):
        model_path = os.path.join(model_path, pt_config['model_file_path'])
        model_class_package = pt_config['model_class_package']
        module = importlib.import_module(model_class_package)
        self._set_sys_modules(model_class_package, module)
        return torch.load(model_path)


    def load_from_saveddict(self, model_path, pt_config):
        model_class_package = pt_config['model_class_package']
        model_class_name = pt_config['model_class_name']
        model_class_init_args = os.path.join(model_path, pt_config['model_class_init_args'])
        model_fullpath = os.path.join(model_path, pt_config['model_file_path'])
        module = importlib.import_module(model_class_package)
        model_class = getattr(module, model_class_name)
        print(f'model_class_init_args = {model_class_init_args}')
        with open(model_class_init_args, 'rb') as fp:
            args = pickle.load(fp)
        return model_class(*args)
        

    def _set_sys_modules(self, package, module):
        entries = package.split('.')
        for i in range(len(entries)):
            sys.modules['.'.join(entries[i:])] = module
            print(f"{'.'.join(entries[i:])} : {sys.modules['.'.join(entries[i:])]}")


class PytorchWrapper(object):
    def __init__(self, model, is_gpu):
        self.model = model
        self.model.eval()
        self.device = 'cuda' if is_gpu else 'cpu'
    
    def predict(self, df):
        print(f"Device type = {self.device}")
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
