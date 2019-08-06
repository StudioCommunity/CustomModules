import os
import yaml
import json
from sys import version_info

PYTHON_VERSION = "{major}.{minor}.{micro}".format(major=version_info.major,
                                                  minor=version_info.minor,
                                                  micro=version_info.micro)
_conda_header = """\
name: project_environment
channels:
  - defaults
"""

_extra_index_url = "--extra-index-url=https://test.pypi.org/simple"
_alghost_pip = "alghost==0.0.59"
_azureml_defaults_pip = "azureml-defaults"

# temp solution, would remove later
_data_type_file_name = "data_type.json"
_data_ilearner_file_name = "data.ilearner"
_conda_file_name = "conda.yaml"
_model_spec_file_name = "model_spec.yml"

def _generate_conda_env(path=None, additional_conda_deps=None, additional_pip_deps=None,
                      additional_conda_channels=None, install_alghost=True, install_azureml=True):
    env = yaml.safe_load(_conda_header)
    env["dependencies"] = ["python={}".format(PYTHON_VERSION), "git", "regex"]
    pip_deps = ([_extra_index_url, _alghost_pip] if install_alghost else []) + (
        [_azureml_defaults_pip] if install_alghost else []) + (
        additional_pip_deps if additional_pip_deps else [])      
    if additional_conda_deps is not None:
        env["dependencies"] += additional_conda_deps
    env["dependencies"].append({"pip": pip_deps})
    if additional_conda_channels is not None:
        env["channels"] += additional_conda_channels

    if path is not None:
        with open(path, "w") as out:
            yaml.safe_dump(env, stream=out, default_flow_style=False)
        return None
    else:
        return env


def _save_conda_env(path, conda_env):
    if conda_env is None:
        raise Exception("conda_env is empty")
    if isinstance(conda_env, str) and os.path.isfile(conda_env):
            with open(conda_env, "r") as f:
                conda_env = yaml.safe_load(f)
    if not isinstance(conda_env, dict):
        raise Exception("Could not load conda_env %s" % conda_env)
    with open(os.path.join(path, _conda_file_name), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)


def _generate_model_spec(flavor_name, model_file_name, conda_file_name=_conda_file_name):
    """
    Generate default model spec (TBD)
    :flavor_name
    :model_file_name
    :conda_file_name (optional)
    """
    spec = {
        'flavor' : {
            'framework' : flavor_name
        },
        flavor_name: {
            'model_file_path': model_file_name
        },
        'conda': {
            'conda_file_path': conda_file_name
        },
    }
    return spec

def _save_model_spec(path, flavor_name, model_file_name, conda_file_name=_conda_file_name):
    """
    Save model spec to local (TBD)
    :path
    :flavor_name
    :model_file_name
    :conda_file_name
    """
    spec = _generate_model_spec(flavor_name, model_file_name, conda_file_name)
    with open(os.path.join(path, _model_spec_file_name), 'w') as fp:
        yaml.dump(spec, fp, default_flow_style=False)

def _generate_ilearner_files(path):
    # Dump data_type.json as a work around until SMT deploys
    dct = {
        "Id": "ILearnerDotNet",
        "Name": "ILearner .NET file",
        "ShortName": "Model",
        "Description": "A .NET serialized ILearner",
        "IsDirectory": False,
        "Owner": "Microsoft Corporation",
        "FileExtension": "ilearner",
        "ContentType": "application/octet-stream",
        "AllowUpload": False,
        "AllowPromotion": False,
        "AllowModelPromotion": True,
        "AuxiliaryFileExtension": None,
        "AuxiliaryContentType": None
    }
    with open(os.path.join(path, _data_type_file_name), 'w') as fp:
        json.dump(dct, fp)
    # Dump data.ilearner as a work around until data type design
    with open(os.path.join(path, _data_ilearner_file_name), 'w') as fp:
        fp.writelines('{}')
