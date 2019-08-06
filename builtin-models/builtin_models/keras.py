import os
import yaml
import keras

from builtin_models.environment import _generate_conda_env
from builtin_models.environment import _generate_ilearner_files
from builtin_models.environment import _save_conda_env
from builtin_models.environment import _save_model_spec

FLAVOR_NAME = "keras"
model_file_name = "model.h5"

def _get_default_conda_env():
    import tensorflow as tf

    return _generate_conda_env(
        additional_pip_deps=[
            "keras=={}".format(keras.__version__),
            "tensorflow=={}".format(tf.__version__),
        ])


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

    if conda_env is None:
        conda_env = _get_default_conda_env()
    _save_conda_env(path, conda_env)

    _save_model_spec(path, FLAVOR_NAME, model_file_name)
    _generate_ilearner_files(path) # temp solution, to remove later

    