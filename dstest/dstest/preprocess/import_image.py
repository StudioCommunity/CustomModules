import logging
import os
import json
import yaml
import click
import pandas as pd
from os import walk
import base64
import pyarrow.parquet as pq
from builtin_score import ioutil
from. import datauri_util

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logging.info(f"in {__file__}")
logging.info(f"Load pyarrow.parquet explicitly: {pq}")
logger = logging.getLogger(__name__)

@click.command()
@click.option('--input_path', default="inputs/mnist")
@click.option('--output_path', default="datas/mnist")
def run(input_path, output_path):
  """
  This functions read images in an folder and encode it ans base64. Then save it as csv in output_path.
  """
  import glob
  types = ('**.jpg', '**.png') # the tuple of file types
  files_grabbed = []
  for files in types:
    pattern = os.path.join(input_path,files)
    files_grabbed.extend(glob.glob(pattern))
  
  print(f"Got {len(files_grabbed)} files in folder {input_path}")
  print(files_grabbed)

  df = pd.DataFrame(columns=["filename", "label", "image"])
  for i in range(len(files_grabbed)):
    filename = files_grabbed[i]
    basename = os.path.splitext(os.path.basename(filename))[0]
    label = basename.split('_')[-1]
    image_64_encode = datauri_util.imgfile_to_datauri(filename)
    df.loc[i] = basename, label, image_64_encode

  ioutil.save_parquet(df, output_path, True)

# python -m dstest.preprocess.import_image  --input_path inputs/mnist --output_path datas/mnist
# python -m dstest.preprocess.import_image  --input_path inputs/imagenet --output_path datas/imagenet
if __name__ == '__main__':
    run()
