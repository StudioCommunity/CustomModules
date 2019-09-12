import os

import yaml
import pandas as pd

from . import constants

MODEL_SPEC_FILES = ["model_spec.yml", "model_spec.yaml"]
ONNX_SPEC_FILES = ["model_opt_spec.yml", "model_opt_spec.yaml"]

class BuiltinScoreModule(object):

    def __init__(self, model_path, params={}):
        print(f"BuiltinScoreModule({model_path}, {params})")
        append_score_column_to_output_value_str = params.get(
            constants.APPEND_SCORE_COLUMNS_TO_OUTPUT_KEY, None
        )
        self.append_score_column_to_output = isinstance(append_score_column_to_output_value_str, str) and\
            append_score_column_to_output_value_str.lower() == "true"
        print(f"self.append_score_column_to_output = {self.append_score_column_to_output}")
        print(f'MODEL_FOLDER: {os.listdir(model_path)}')

        model_spec_path = self.get_model_spec_path(model_path)
        with open(model_spec_path) as fp:
            config = yaml.safe_load(fp)
        
        framework = config["flavor"]["framework"]
        if framework.lower() == "pytorch":
            from .pytorch_score_module import PytorchScoreModule
            self.module = PytorchScoreModule(model_path, config)
        elif framework.lower() == "tensorflow":
            from .tensorflow_score_module import TensorflowScoreModule
            self.module = TensorflowScoreModule(model_path, config)
        elif framework.lower() == "sklearn":
            from .sklearn_score_module import SklearnScoreModule
            self.module = SklearnScoreModule(model_path, config)
        elif framework.lower() == "keras":
            from .keras_score_module import KerasScoreModule
            self.module = KerasScoreModule(model_path, config)
        elif framework.lower() == "python":
            from .python_score_module import PythonScoreModule
            self.module = PythonScoreModule(model_path, config)
        elif framework.lower() == "onnx":
            from .onnx_score_module import OnnxScoreModule
            self.module = OnnxScoreModule(model_path, config)
        else:
            msg = f"Not Implemented: framework {framework} not supported"
            print(msg)
            raise ValueError(msg)
        
        self.onnx = None
        onnx_spec_path = self.get_onnx_spec_path(model_path)
        if onnx_spec_path:
            from .onnx_score_module import OnnxScoreModule
            with open(onnx_spec_path) as fp:
                config = yaml.safe_load(fp)
            self.onnx = OnnxScoreModule(model_path, config)
        print(f'ONNX={self.onnx}, PATH={onnx_spec_path}')

    def run(self, df, global_param=None):
        output_label = pd.DataFrame()
        if self.onnx:
            try:
                print(f"Start ONNX run")
                output_label = self.onnx.run(df)
                print(f"End ONNX run")
            except Exception as ex:
                print(f"ONNX EXCEPTION: {ex}")
                print(f"Fallback to the original model")
        if output_label.empty:
            output_label = self.module.run(df)
        if self.append_score_column_to_output:
            if isinstance(output_label, pd.DataFrame):
                df = pd.concat([df, output_label], axis=1)
            else:
                df.insert(len(df.columns), constants.SCORED_LABEL_COL_NAME, output_label, True)
        else:
            if isinstance(output_label, pd.DataFrame):
                df = output_label
            else:
                df = pd.DataFrame({constants.SCORED_LABEL_COL_NAME: output_label})
        print(f"df =\n{df}")
        print(df.columns)
        if df.shape[0] > 0:
            for col in df.columns:
                print(f"{col}: {type(df.loc[0][col])}")
        return df

    def get_model_spec_path(self, model_path):
        filenames = []
        for filename in MODEL_SPEC_FILES:
            model_spec_path = os.path.join(model_path, filename)
            filenames.append(model_spec_path)
            if os.path.exists(model_spec_path):
                return model_spec_path
        raise FileNotFoundError(str(filenames))


    def get_onnx_spec_path(self, model_path):
        for filename in ONNX_SPEC_FILES:
            onnx_spec_path = os.path.join(model_path, filename)
            if os.path.exists(onnx_spec_path):
                return onnx_spec_path
        return ''
