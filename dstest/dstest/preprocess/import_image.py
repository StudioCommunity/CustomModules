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

  df = pd.DataFrame(columns=["label", "image"])
  for i in range(len(files_grabbed)):
    filename = files_grabbed[i]
    label = os.path.splitext(os.path.basename(filename))[0].split('_')[-1]
    with open(filename, 'rb') as image:
      image_read = image.read()
      image_64_encode = base64.encodebytes(image_read).decode('ascii')
      df.loc[i] = label, image_64_encode

  ioutil.save_parquet(df, output_path)

# python -m dstest.preprocess.import_image  --input_path inputs/mnist --output_path datas/mnist
if __name__ == '__main__':
    run()
