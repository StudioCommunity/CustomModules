import skimage
import skimage.io
import skimage.transform
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def load_image(path, target_size = (224,224)):
  # load image
  img = skimage.io.imread(path)
  img = img/ 255.0
  assert (0 <= img).all() and (img <= 1.0).all()
  # we crop image from center
  short_edge = min(img.shape[:2])
  
  yy = int((img.shape[0] - short_edge) / 2)
  xx = int((img.shape[1] - short_edge) / 2)
  crop_img = img[yy : yy + short_edge, xx : xx + short_edge]
  # resize to targetSize
  resized_img = skimage.transform.resize(crop_img, target_size)
  return resized_img

# convert this to a generic postprocess module
synset = [l.strip() for l in open('model/vgg/synset.txt').readlines()]
def get_category(prob):
  pred = np.argsort(prob)[::-1] ##[::-1] inverse order
  # Get top1 label
  top1 = synset[pred[0]]
  # Get top5 label
  #top5 = [synset[pred[i]] for i in range(5)]
  return top1


def load_model():
  # VGG-16: https://github.com/ry/tensorflow-vgg16
  # \\clement-pc1\share\clwan\model\vgg16-model files
  with open("model/vgg/vgg16-20160129.tfmodel", mode='rb') as f:
    fileContent = f.read()
  ##定义输入
  #images = tf.placeholder("float", [None, 224, 224, 3])
  graph_def = tf.GraphDef()
  graph_def.ParseFromString(fileContent)
  tf.import_graph_def(graph_def)
  graph = tf.get_default_graph()
  ##force cpu mode
  config = tf.ConfigProto(
    device_count = {'GPU': 0}
  )

  sess = tf.Session(config=config)
  init = tf.global_variables_initializer()
  sess.run(init)
  return graph, sess

def load_inputs():
  imgs=[]
  for i in ['cat.jpg', "dog1.jpg"]:
    img = load_image('inputs/imagenet/'+i)
    #skimage.io.imsave(f"outputs/{i}-transform.jpg", img)
    #plt.imshow(img)
    #plt.show()
    imgs.append(img)
  return imgs

# python -m dstest.tensorflow.vgg.loadvgg
if __name__ == '__main__':
  graph, sess = load_model()

  imgs = load_inputs()
  img_num=len(imgs)

  batch = np.array(imgs).reshape((img_num, 224, 224, 3))
  assert batch.shape == (img_num, 224, 224, 3)
  images = graph.get_tensor_by_name("import/images:0")
  prob_tensor = graph.get_tensor_by_name("import/prob:0")
  feed_dict = { images: batch }
  prob = sess.run(prob_tensor, feed_dict=feed_dict)
  print("here")
  print(prob.shape)

  result=[]
  for i in range (img_num): 
    category = get_category(prob[i])
    result.append(category)
  
  print ("The category is :",result)
  sess.close()
