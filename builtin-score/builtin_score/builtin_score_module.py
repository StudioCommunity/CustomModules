import os

import yaml
import pandas as pd

from . import constants

MODEL_SPEC_FILE_NAME = "model_spec.yml"


class BuiltinScoreModule(object):

    def __init__(self, model_path, params={}):
        print(f"BuiltinScoreModule({model_path}, {params})")
        append_score_column_to_output_value_str = params.get(
            constants.APPEND_SCORE_COLUMNS_TO_OUTPUT_KEY, None
        )
        self.append_score_column_to_output = isinstance(append_score_column_to_output_value_str, str) and\
            append_score_column_to_output_value_str.lower() == "true"
        print(f"self.append_score_column_to_output = {self.append_score_column_to_output}")
        model_spec_path = os.path.join(model_path, MODEL_SPEC_FILE_NAME)
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
        else:
            print(f"Not Implemented: framework {framework} not supported")

    def run(self, df, global_param=None):
        output_label = self.module.run(df)
        if self.append_score_column_to_output:
            if isinstance(output_label, pd.DataFrame):
                return pd.concat([df, output_label], axis=1)
            else:
                df.insert(len(df.columns), constants.SCORED_LABEL_COL_NAME, output_label, True)
        else:
            if isinstance(output_label, pd.DataFrame):
                df = output_label
            else:
                df = pd.DataFrame({constants.SCORED_LABEL_COL_NAME: output_label})
        print(df)
        return df
