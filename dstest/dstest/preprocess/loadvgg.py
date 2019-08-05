import skimage
import skimage.transform
import numpy as np
import pandas as pd
import tensorflow as tf
from . import imagenet
from builtin_score import tensorflow_score_module
from builtin_score.tensorflow_score_module import TensorflowScoreModule

def load_image(path, target_size = (224,224)):
  img = skimage.io.imread(path)
  transformed = imagenet.transform_image_imagenet(img, target_size)
  return transformed

# convert this to a generic postprocess module
synset = [l.strip() for l in open('model/vgg/synset.txt').readlines()]
def get_category(prob):
  pred = np.argsort(prob)[::-1] ##[::-1] inverse order
  # Get top1 label
  top1 = synset[pred[0]]
  # Get top5 label
  #top5 = [synset[pred[i]] for i in range(5)]
  return top1

model_path = "model/vgg/"

def load_model(sess):
  # VGG-16: https://github.com/ry/tensorflow-vgg16
  # \\clement-pc1\share\clwan\model\vgg16-model files
  return tensorflow_score_module.load_graph("model/vgg/vgg16-20160129.tfmodel", sess)

def load_inputs():
  imgs=[]
  for i in ['cat.jpg', "dog1.jpg", "dog2.jpg"]:
    img = load_image('inputs/imagenet/'+i)
    skimage.io.imsave(f"outputs/{i}-transform.jpg", img)
    img = img / 255.0
    img = img.flatten()
    imgs.append(img)
  df = pd.DataFrame()
  df.insert(len(df.columns), "import/images", imgs, True)
  return df

# python -m dstest.tensorflow.vgg.loadvgg
if __name__ == '__main__':
  df = load_inputs()
  print(df)
  
  import yaml
  with open(model_path + "model_spec.yml") as fp:
      config = yaml.safe_load(fp)

  tfmodule = TensorflowScoreModule(model_path, config)
  result_df = tfmodule.run(df)
  print(result_df)
  prob = result_df["import/prob"]

  result=[]
  for i in range (len(result_df)): 
    category = get_category(prob[i])
    result.append(category)
  
  print ("The category is :",result)
