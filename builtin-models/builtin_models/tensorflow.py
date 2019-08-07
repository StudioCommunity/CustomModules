import os
import yaml
import cloudpickle
import tensorflow

from builtin_models.environment import _generate_conda_env
from builtin_models.environment import _generate_ilearner_files

FLAVOR_NAME = "tensorflow"
model_file_name = "model.meta"
conda_file_name = "conda.yaml"
model_spec_file_name = "model_spec.yml"


def _get_default_conda_env():
    return utils.generate_conda_env(
        additional_pip_deps=[
            "tensorflow=={}".format(tensorflow.__version__)
        ])


def _save_model_spec(path, model):
    spec = dict(
        flavor = dict(
            framework = FLAVOR_NAME
        ),
        tensorflow = dict(
            model_file_path = model_file_name,
            inputs = dict(),
            outputs = dict()
        ),
        conda = dict(
            conda_file_path = conda_file_name
        )
    )
    # TO DO
    # inputs outputs setting, read from model
    with open(os.path.join(path, model_spec_file_name), 'w') as fp:
        yaml.dump(spec, fp, default_flow_style=False)


def _save_model(model, path):
    with open(path, 'wb') as fp:
        cloudpickle.dump(model, fp)


def save_model(path='./model/', tf_meta_graph_tags, tf_signature_def_key, conda_env=None):
    """
    Save a Tensorflow model to a path on the local file system.

    :param pytorch_model: PyTorch model to be saved. 

    :param path: Path to a directory containing model data.

    :param conda_env: Either a dictionary representation of a Conda environment or the path to a conda environment yaml file. 
    """
    if(not path.endswith('/')):
        path += '/'
    if not os.path.exists(path):
        os.makedirs(path)

    _save_model(tensorflow_model, os.path.join(path, model_file_name))
    _save_conda_env(path, conda_env)
    _save_model_spec(path, tensorflow_model)
    _generate_ilearner_files(path) # temp solution, to remove later

    