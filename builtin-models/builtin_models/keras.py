import os
import yaml
import keras

from builtin_models.environment import _generate_conda_env
from builtin_models.environment import _generate_ilearner_files

FLAVOR_NAME = "keras"
model_file_name = "model.h5"
conda_file_name = "conda.yaml"
model_spec_file_name = "model_spec.yml"

def _get_default_conda_env():
    import tensorflow as tf

    return _generate_conda_env(
        additional_pip_deps=[
            "keras=={}".format(keras.__version__),
            "tensorflow=={}".format(tf.__version__),
        ])


def _save_conda_env(path, conda_env=None):
    if conda_env is None:
        conda_env = _get_default_conda_env()
    elif not isinstance(conda_env, dict):
        with open(conda_env, "r") as f: # conda_env is a file
            conda_env = yaml.safe_load(f)
    with open(os.path.join(path, conda_file_name), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)


def _save_model_spec(path):
    spec = {
        'flavor' : {
            'framework' : FLAVOR_NAME
        },
        FLAVOR_NAME: {
            'model_file_path': model_file_name
        },
        'conda': {
            'conda_file_path': conda_file_name
        },
    }
    with open(os.path.join(path, model_spec_file_name), 'w') as fp:
        yaml.dump(spec, fp, default_flow_style=False)


def _load_model_from_local_file(path):
    from keras.models import load_model
    return load_model(path)


def save_model(keras_model, path='./model/', conda_env=None):
    """
    Save a Keras model to a path on the local file system.

    :param keras_model: Keras model to be saved. 

    :param path: Path to a file or directory containing model data.
    
    :param conda_env: Either a dictionary representation of a Conda environment or the path to a conda environment yaml file. 
    """
    if(not path.endswith('/')):
        path += '/'
    if not os.path.exists(path):
        os.makedirs(path)

    keras_model.save(os.path.join(path, model_file_name)) 
    _save_conda_env(path, conda_env)
    _save_model_spec(path)
    _generate_ilearner_files(path) # temp solution, to remove later

    