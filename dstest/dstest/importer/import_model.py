import os
import os.path
import sys
import json
import importlib
import imp

from pip._internal import main as pipmain
pipmain(["install", "click"])
import click

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logging.info(f"Import Model")
logger = logging.getLogger(__name__)

class Importer(object):
    def __init__(self, input_path, flavor, out_model_path):
        self.input_path = input_path
        self.flavor = flavor
        self.out_model_path = out_model_path
        self.modules = {}
        self.dependencies = []

    def run(self, model_file, serialization_mode, init_args):
        self.load_modules()
        if self.flavor == 'pytorch':
            self.load_pytorch(model_file, serialization_mode, init_args)
        elif self.flavor == 'keras':
            self.load_keras(model_file, serialization_mode)
        elif self.flavor == 'tensorflow':
            self.load_tensorflow(model_file, serialization_mode)
        elif self.flavor == 'sklearn':
            self.load_sklearn(model_file, serialization_mode)
        else:
            raise NotImplementedError

    def extract_filename(self, url):
        return url.partition('?')[0].rpartition('/')[-1]

    def parse_init_args(self, init_args):
        if not init_args:
            return '', {}
        init_args = init_args.replace("'", '"').replace(";",",")
        print(f'INIT_ARGS: {init_args}')
        args = json.loads(init_args)
        class_name = args.get('class', '')
        if class_name:
            args.pop('class')
        return class_name, args

    def load_modules(self):
        print(f'INPUT_PATH: {self.input_path}')
        print(f'INPUT_PATH_FILES: {os.listdir(self.input_path)}')
        for basedir, dirs, files in os.walk(self.input_path):
            for fn in files:
                if fn.endswith('.py') and fn != 'setup.py':
                    name = fn[:-len('.py')]
                    py_file = os.path.join(basedir, fn)
                    print(f'LOAD {name} from {py_file}')
                    self.modules[name] = imp.load_source(name, py_file)
                    self.dependencies.append(py_file)
        
    def load_pytorch(self, model_file, serialization_mode, init_args):
        import torch
        from builtin_models.pytorch import save_model
        dependencies = []
        model = None
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'DEVICE={device}')
        if not os.path.exists(model_file):
            model_file = os.path.join(self.input_path, model_file)
        if serialization_mode == 'cloudpickle':
            print(f'CLOUDPICKLE: {model_file} to {self.out_model_path}')
            import cloudpickle
            retry = True
            while retry:
                try:
                    with open(model_file, 'rb') as fp:
                        model = cloudpickle.load(fp)
                    retry = False
                except ModuleNotFoundError as ex:
                    name = ex.name.rpartition('.')[-1]
                    if name in modules:
                        sys.modules[ex.name] = modules[name]
                        retry = True
                    else:
                        raise ex      
        elif serialization_mode == 'savedmodel':
            print(f'SAVEDMODEL: {model_file} to {self.out_model_path}')
            retry = True
            while retry:   
                try:
                    with open(model_file, 'rb') as fp:
                        model = torch.load(model_file, map_location=device)
                    retry = False
                except ModuleNotFoundError as ex:
                    name = ex.name.rpartition('.')[-1]
                    if name in modules:
                        sys.modules[ex.name] = modules[name]
                        retry = True
                    else:
                        raise ex
        elif serialization_mode == 'statedict':
            print(f'STATEDICT: {model_file} to {self.out_model_path}')
            class_name, init_args = self.parse_init_args(init_args)
            if not class_name:
                if len(self.modules) == 1:
                    class_name = list(self.modules.keys())[0]
                else:
                    raise NotImplementedError
            print(f'CLASS_NAME={class_name}')
            model_class = None
            for module in self.modules.values():
                model_class = getattr(module, class_name, None)
                if model_class:
                    break
            if not model_class:
                raise NotImplementedError

            print(f'INIT = {init_args}')
            if init_args:
                model = model_class(**init_args)
            else:
                model = model_class()
            model.load_state_dict(torch.load(model_file, map_location=device))
        else:
            raise NotImplementedError

        save_model(model, self.out_model_path, dependencies=self.dependencies)
        print(f'OUT_MODEL_FOLDER: {os.listdir(self.out_model_path)}')
    def load_keras(self, model_file, serialization_mode):
        pass
    def load_tensorflow(self, model_file, serialization_mode):
        pass
    def load_keras(self, model_file, serialization_mode):
        pass
        

@click.command()
@click.option('--input_path')
@click.option('--flavor')
@click.option('--model_file')
@click.option('--serialization_mode')
@click.option('--init_args', default='{}')
@click.option('--out_model_path', default='model')
def run_pipeline(input_path, flavor, model_file, serialization_mode, init_args, out_model_path):
    importer = Importer(input_path, flavor, out_model_path)
    importer.run(model_file, serialization_mode, init_args)

# python -m dstest.importer.import_model --input_path download --flavor pytorch --model_file 200000-G.ckpt --serialization_mode statedict --init_args "{'class':'Generator','conv_dim':64,'c_dim':5,'repeat_num':6}"
# python -m dstest.importer.import_model --input_path download --flavor pytorch --model_file model.pt --serialization_mode savedmodel
# python -m dstest.importer.import_model --input_path download --flavor pytorch --model_file model.pkl --serialization_mode cloudpickle
if __name__ == '__main__':
    run_pipeline()