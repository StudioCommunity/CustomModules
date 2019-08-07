import os
import yaml
import keras
import builtin_models.utils as utils

FLAVOR_NAME = "keras"
model_file_name = "model.h5"

def _get_default_conda_env():
    import tensorflow as tf
    return utils.generate_conda_env(
        additional_pip_deps=[
            "keras=={}".format(keras.__version__),
            "tensorflow=={}".format(tf.__version__),
        ])


def save_model(keras_model, path='./model/', conda_env=None):
    """
    Save a Keras model to a path on the local file system.

    :param keras_model: Keras model to be saved. 

    :param path: Path to a directory containing model data.
    
    :param conda_env: Either a dictionary representation of a Conda environment or the path to a conda environment yaml file. 
    """
    if(not path.endswith('/')):
        path += '/'
    if not os.path.exists(path):
        os.makedirs(path)

    keras_model.save(os.path.join(path, model_file_name)) 

    if conda_env is None:
        conda_env = _get_default_conda_env()
    utils.save_conda_env(path, conda_env)

    utils.save_model_spec(path, FLAVOR_NAME, model_file_name)
    utils.generate_ilearner_files(path) # temp solution, to remove later

    