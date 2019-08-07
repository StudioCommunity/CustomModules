import os
import yaml
import cloudpickle
import torch
import torchvision

import builtin_models.utils as utils

FLAVOR_NAME = "pytorch"
model_file_name = "model.pkl"

def _get_default_conda_env():
    return utils.generate_conda_env(
        additional_pip_deps=[
            "torch=={}".format(torch.__version__),
            "torchvision=={}".format(torchvision.__version__),
        ])


def _save_model(pytorch_model, path):
    with open(path, 'wb') as fp:
        cloudpickle.dump(pytorch_model, fp)


def save_model(pytorch_model, path='./model/', conda_env=None):
    """
    Save a PyTorch model to a path on the local file system.

    :param pytorch_model: PyTorch model to be saved. 

    :param path: Path to a directory containing model data.

    :param conda_env: Either a dictionary representation of a Conda environment or the path to a conda environment yaml file. 
    """
    if(not path.endswith('/')):
        path += '/'
    if not os.path.exists(path):
        os.makedirs(path)

    # only save cpu version
    _save_model(pytorch_model.to('cpu'), os.path.join(path, model_file_name))

    if conda_env is None:
        conda_env = _get_default_conda_env()
    utils.save_conda_env(path, conda_env)

    utils.save_model_spec(path, FLAVOR_NAME, model_file_name)
    utils.generate_ilearner_files(path) # temp solution, to remove later


    