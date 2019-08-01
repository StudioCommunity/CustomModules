import os
import yaml
import pickle

from builtin_models.environment import _generate_conda_env
from builtin_models.environment import _generate_ilearner_files

FLAVOR_NAME = "sklearn"
model_file_name = "model.pkl"
conda_file_name = "conda.yaml"
model_spec_file_name = "model_spec.yml"

def _get_default_conda_env():
    import sklearn

    return _generate_conda_env(
        additional_pip_deps=[
            "scikit-learn=={}".format(sklearn.__version__)
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


def _save_model(sklearn_model, path):
    with open(os.path.join(path, model_file_name), "wb") as fb:
        pickle.dump(sklearn_model, fb)


def load_model_from_local_file(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_model(sklearn_model, path, conda_env=None):
    import sklearn

    if(not path.endswith('/')):
        path += '/'
    if not os.path.exists(path):
        os.makedirs(path)

    _save_model(sklearn_model, path)
    _save_conda_env(path, conda_env)
    _save_model_spec(path)
    _generate_ilearner_files(path) # temp solution, to remove later

    