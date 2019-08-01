import os
import mimetypes
import base64
import cv2
import numpy as np

def remove_datauri_prefix(data_uri):
    """Remove prefix of a data URI to base64 content."""
    return data_uri.split(',')[-1]


def img_to_datauri(img):
    """
    convert 
    data : cv2 image Mat
    """
    data = cv2.imencode('.jpg', img)[1].tobytes()
    filetype = "jpg"
    data64 = u''.join(base64.encodebytes(data).decode('ascii').splitlines())
    #cv2.imwrite('outputs/test_3.png',gray)
    return u'data:image/%s;base64,%s' % (filetype, data64)

def imgfile_to_data(filename):
    with open(filename, 'rb') as image:
      image_read = image.read()
    #   image_64_encode = base64.encodebytes(image_read).decode('ascii')
      image_64_encode = u''.join(base64.encodebytes(image_read).decode('ascii').splitlines())
    return image_64_encode

def imgfile_to_datauri(filename):
    """Convert a file (specified by a  filename) into a data URI."""
    if not os.path.exists( filename):
        raise FileNotFoundError
    mime, _ = mimetypes.guess_type( filename)
    with open( filename, 'rb') as fp:
        data = fp.read()
        data64 = u''.join(base64.encodebytes(data).decode('ascii').splitlines())
        return u'data:%s;base64,%s' % (mime, data64)

def readb64(base64_string):
  imgData = base64.b64decode(base64_string)
  nparr = np.fromstring(imgData, np.uint8)
  img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
  #print(img.shape)
  #cv2.imwrite("outputs/test_0_origin.png", img)
  return img

def _write_file(str, filename):
  with open(filename, "w") as fp:
    fp.write(str)

# python -m dstest.preprocess.datauri_util
if __name__ == '__main__':
  uri = imgfile_to_datauri("inputs/mnist/1.jpg")
  #print(uri)
  img = readb64(remove_datauri_prefix(uri))
  cv2.imwrite("outputs/test1.jpg", img)
  #print(img)
  uri1 = img_to_datauri(img)
  # print(img)
  print(uri1 == uri)
  _write_file(uri1, "uri1.txt")
  _write_file(uri, "uri.txt")
  img = readb64(remove_datauri_prefix(uri1))
  cv2.imwrite("outputs/test.jpg", img)