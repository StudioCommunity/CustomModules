import os
import shutil
import yaml
import cloudpickle
import inspect
import builtin_models.utils as utils

FLAVOR_NAME = "python"
MODEL_FILE_NAME = "model.pkl"

class PythonModel(object):
    """
    Represents a generic Python model that evaluates inputs and produces API-compatible outputs.
    By subclassing :class:`~PythonModel`, users can create customized models.
    """
    # def __init__(self, model_path):
    #     """
    #     Loads model from model_path.

    #     :param context: model_path containing artifacts that the model can use to perform inference.
    #     """

    def predict(self, *args):
        """
        Evaluates a pyfunc-compatible input and produces a pyfunc-compatible output.

        :param args: variable number of input arguments.

        """

def _get_default_conda_env():
    return utils.generate_conda_env(
        additional_pip_deps=[
            "cloudpickle=={}".format(cloudpickle.__version__),
        ])


def _save_model(python_model, path):
    with open(path, 'wb') as fp:
        cloudpickle.dump(python_model, fp)


def save_model(python_model, path='./model/', conda_env=None, dependencies=[]):
    """
    Save a generic python model to a path on the local file system.

    :param python_model: Python model to be saved. 

    :param path: Path to a directory containing model data.

    :param conda_env: Either a dictionary representation of a Conda environment or the path to a conda environment yaml file. 
    """
    if(not path.endswith('/')):
        path += '/'
    if not os.path.exists(path):
        os.makedirs(path)

    # only save cpu version
    _save_model(python_model, os.path.join(path, MODEL_FILE_NAME))
    fn = os.path.join(path, MODEL_FILE_NAME)
    print(f'MODEL_FILE: {fn}')
    
    if conda_env is None:
        conda_env = _get_default_conda_env()
    print(f'path={path}, conda_env={conda_env}')
    utils.save_conda_env(path, conda_env)

    for dependency in dependencies:
        shutil.copy(dependency, path)
    forward_func = getattr(python_model, 'predict')
    args = inspect.getargspec(forward_func).args
    if 'self' in args:
        args.remove('self')

    utils.save_model_spec(path, FLAVOR_NAME, MODEL_FILE_NAME, input_args = args)
    utils.generate_ilearner_files(path) # temp solution, to remove later

def load_model(path):
    model_file_path = os.path.join(path, MODEL_FILE_NAME)
    with open(model_file_path, 'rb') as fp:
        model = cloudpickle.load(fp)
    return model