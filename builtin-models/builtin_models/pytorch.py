import os
import yaml
import cloudpickle
import torch
import torchvision

from builtin_models.environment import _generate_conda_env
from builtin_models.environment import _generate_ilearner_files
from builtin_models.environment import _save_conda_env
from builtin_models.environment import _generate_model_spec

FLAVOR_NAME = "pytorch"
model_file_name = "model.pkl"
gpu_model_file_name = "cuda_model.pkl"
model_spec_file_name = "model_spec.yml"


def _get_default_conda_env():
    return _generate_conda_env(
        additional_pip_deps=[
            "torch=={}".format(torch.__version__),
            "torchvision=={}".format(torchvision.__version__),
        ])


def _save_model_spec(path, isGpu = False):
    spec = _generate_model_spec(FLAVOR_NAME, model_file_name)
    if isGpu:
        spec[FLAVOR_NAME]['cuda_model_file_path'] = gpu_model_file_name
    with open(os.path.join(path, model_spec_file_name), 'w') as fp:
        yaml.dump(spec, fp, default_flow_style=False)


def _save_model(pytorch_model, path):
    with open(path, 'wb') as fp:
        cloudpickle.dump(pytorch_model, fp)


def _load_model_from_local_file(path):
    with open(path, 'rb') as fp:
        model = cloudpickle.load(fp)
    return model


def save_model(pytorch_model, path='./model/', conda_env=None):
    """
    Save a PyTorch model to a path on the local file system.

    :param pytorch_model: PyTorch model to be saved. 

    :param path: Path to a file or directory containing model data.

    :param conda_env: Either a dictionary representation of a Conda environment or the path to a conda environment yaml file. 
    """
    if(not path.endswith('/')):
        path += '/'
    if not os.path.exists(path):
        os.makedirs(path)

    is_gpu = torch.cuda.is_available()
    # save gpu version
    if is_gpu:
        _save_model(pytorch_model.to('cuda'), os.path.join(path, gpu_model_file_name))
    # save cpu version too
    _save_model(pytorch_model.to('cpu'), os.path.join(path, model_file_name))

    if conda_env is None:
        conda_env = _get_default_conda_env()
    _save_conda_env(path, conda_env)

    _save_model_spec(path, is_gpu)
    _generate_ilearner_files(path) # temp solution, to remove later

    