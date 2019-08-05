import logging
import click
import pandas as pd
import cv2
from builtin_score import ioutil
from . import datauri_util
from . import mnist

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logging.info(f"in {__file__} v1")
logger = logging.getLogger(__name__)

def add_data_to_dataframe(input_df, results, target_column):
  if(input_df.columns.contains(target_column)):
    logger.info(f"writing to column {target_column}")
    input_df[target_column] = results
  else:
    logger.info(f"append new column {target_column}")
    input_df.insert(len(input_df.columns), target_column, results, True)

class PreProcess:
  def __init__(self, meta: dict = {}):
    self.image_column = str(meta.get('Image Column', 'image'))
    self.target_column = str(meta.get('Target Column', ''))
    self.target_data_column = str(meta.get('Target DataURI Column', ''))
    
    if not self.target_column:
      self.target_column = self.image_column

    if not self.target_data_column:
      self.target_data_column = f"{self.target_column}_data"

    print(self.image_column, self.target_column)

  def run(self, input_df: pd.DataFrame, meta: dict = None):
    results = []
    datauris = []
    
    for index, row in input_df.iterrows():
      #print(row['label'])
      img = datauri_util.base64str_to_ndarray(row[self.image_column])
      img = mnist.transform_image_mnist(img)
      
      # append datauris
      if self.target_data_column:
        datauri = datauri_util.img_to_datauri(img)
        datauris.append(datauri)
      
      # save the processed images
      cv2.imwrite("outputs/image_"+str(index)+".png", img)
      
      # Convert to 0-1 based range, so we can save it in dataframe
      flatten = img.flatten() / 255.0
      results.append(flatten)

    add_data_to_dataframe(input_df, results, self.target_column)
    if self.target_data_column:
      add_data_to_dataframe(input_df, datauris, self.target_data_column)

    return input_df

@click.command()
@click.option('--input_path', default="datas/mnist")
@click.option('--output_path', default="outputs/mnist")
@click.option('--image_column', default="image")
@click.option('--target_column', default="x")
@click.option('--target_datauri_column', default="")
def run(input_path, output_path, image_column, target_column, target_datauri_column):
  """
  This functions read base64 encoded images from df. Transform to format required by model input.
  """
  meta = {
    "Image Column": image_column,
    "Target Column": target_column,
    "Target DataURI Column": target_datauri_column
  }
  proccesor = PreProcess(meta)

  df = ioutil.read_parquet(input_path)
  result = proccesor.run(df)
  ioutil.save_parquet(result, output_path)

# python -m dstest.preprocess.preprocess  --input_path datas/mnist --output_path outputs/mnist --image_column=image --target_column=x --target_datauri_column=x.data
if __name__ == '__main__':
  run()
