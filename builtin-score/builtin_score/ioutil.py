
import logging
import os
import json
import pandas as pd
import numpy as np

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logging.info(f"in {__file__}")
logger = logging.getLogger(__name__)

def read_parquet(data_path):
    """
    :param file_name: str,
    :return: pandas.DataFrame
    """
    logger.info("start reading parquet.")
    df = pd.read_parquet(os.path.join(data_path, 'data.dataset.parquet'), engine='pyarrow')
    logger.info("parquet read completed.")
    return df

def ensure_folder_exists(output_path):
  if not os.path.exists(output_path):
    logger.info(f"{output_path} not exists")
    os.makedirs(output_path)

def save_parquet(df, output_path):
  ensure_folder_exists(output_path)
  #df.to_csv(os.path.join(output_path, "data.csv"))

  df.to_parquet(fname=os.path.join(output_path, "data.dataset.parquet"), engine='pyarrow')

  # Dump data_type.json as a work around until SMT deploys
  dct = {
      "Id": "Dataset",
      "Name": "Dataset .NET file",
      "ShortName": "Dataset",
      "Description": "A serialized DataTable supporting partial reads and writes",
      "IsDirectory": False,
      "Owner": "Microsoft Corporation",
      "FileExtension": "dataset.parquet",
      "ContentType": "application/octet-stream",
      "AllowUpload": False,
      "AllowPromotion": True,
      "AllowModelPromotion": False,
      "AuxiliaryFileExtension": None,
      "AuxiliaryContentType": None
  }
  with open(os.path.join(output_path, 'data_type.json'), 'w') as f:
    json.dump(dct, f)

def from_df_column_to_array(col):
  if(len(col)==0):
    return []
  
  if(col.dtype == 'O'):
    shape = []
    shape.append(len(col))
    shape.append(len(col[0]))
    values = np.zeros(shape)
    for i in range(len(col)):
      values[i] = col[i]
    return values
  else:
    return col.values
  