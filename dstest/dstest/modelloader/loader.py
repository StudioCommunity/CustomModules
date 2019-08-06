import os
import logging
import keras
import urllib.request

from pip._internal import main as pipmain
pipmain(["install", "click"])
import click

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logging.info(f"in dstest echo")
logger = logging.getLogger(__name__)

@click.command()
@click.option('--flavor', default='pytorch')
@click.option('--model_url', default='model.pkl')
@click.option('--serialization', default='cloudpickle')
@click.option('--out_model_path', default='model')
def run_pipeline(flavor, model_url, serialization, out_model_path):
    download_path = 'download'
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    model_file = os.path.join(download_path, 'model.file')
    urllib.request.urlretrieve(model_url, model_file)
    if flavor == 'pytorch':
        load_pytorch(model_file, serialization, out_model_path)
    elif flavor == 'keras':
        load_keras(model_file, serialization, out_model_path)
    elif flavor == 'tensorflow':
        load_tensorflow(model_file, serialization, out_model_path)
    elif flavor == 'sklearn':
        load_sklearn(model_file, serialization, out_model_path)
    else:
        raise NotImplementedError()

def load_pytorch(model_file, serialization, out_model_path):
    from builtin_models.pytorch import save_model
    if serialization == 'cloudpickle':
        print(f'model loading(cloudpickle): {model_file} to {out_model_path}')
        import cloudpickle
        with open(model_file, 'rb') as fp:
            model = cloudpickle.load(fp)
        save_model(model, out_model_path)
        print(f'model loaded: {out_model_path}')
    elif serialization == 'savedmodel':
        pass
    else:
        pass

def load_keras(model_file, serialization, out_model_path):
    from builtin_models.keras import save_model, load_model_from_local_file
    model = load_model_from_local_file(model_file)
    path = './model'
    save_model(model, path)

def load_tensorflow(model_file, serialization, out_model_path):
    if serialization == 'saved_model':
        pass
    elif serialization == 'saver':
        pass
    else:
        pass

def load_sklearn(model_file, serialization, out_model_path):
    if serialization == 'pickle':
        pass
    else:
        pass

# python -m dstest.modelloader.loader --flavor pytorch --model_url "https://zhweiamlservic0807301744.blob.core.windows.net/models/model.pkl?sp=r&st=2019-08-06T05:28:37Z&se=2019-08-31T13:28:37Z&spr=https&sv=2018-03-28&sig=naVtXXnxEZiGKWyOo8Vst35tekpmMmLwEKRD7FNUhwI%3D&sr=b" --serialization cloudpickle --output_model_path model2
if __name__ == '__main__':
    run_pipeline()
