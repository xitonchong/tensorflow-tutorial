import re
import tensorflow as tf
import glob
import numpy as np
import re 
import time
import os
from scipy import misc
import imageio
# this only works for tf 1.13
print(tf.__version__)
tf.enable_eager_execution()

img_paths = "/data/DIV2K"
img_train = [ "DIV2K_train_LR_unknown/X4", "DIV2K_valid_LR_unknown/X4"] 
img_label = ["DIV2K_train_HR" , "DIV2K_valid_HR"]

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def load_image(img_name):
    img = imageio.imread(img_name)
    return img

def serialize_dataset(train, valid):
    feature = {
            'train': _bytes_feature(train),
            #tf.serialize_tensor(tf.cast(train, tf.float32))),
            'label': _bytes_feature(valid)
            # tf.serialize_tensor(tf.cast(valid, tf.float32)))
    }
    #tf_string = tf.serialize_tensor(tf.cast(value, tf.float32))
    #return tf.reshape(tf_string, ())
    example_proto = tf.train.Example(
            features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

start_time = time.time()

for i in range(len(img_train)):
    print(i)
    img_train_path = os.path.join(img_paths, img_train[i])
    img_label_path = os.path.join(img_paths, img_label[i])
    img_train_list = glob.glob(img_train_path + "/*.png")
    img_label_list = glob.glob(img_label_path + "/*.png")
    img_train_list.sort()
    img_label_list.sort()
    print(img_train[i], img_train_list[0:5])
    print(img_label[i], img_label_list[0:5])
    for j in range(len(img_train_list)):
        #train = load_image(img_train_list[j])
        #label = load_image(img_label_list[j])
        train = open(img_train_list[j], 'rb').read()
        label = open(img_label_list[j], 'rb').read()
        #my_feature_dataset = tf.data.Dataset.from_tensor_slices((train,label))
        serialized_example = serialize_dataset(train, label)
        #my_feature_dataset
        #my_serialized_dataset = my_feature_dataset.map(serialize_dataset)
        # remove .png extension to .tfrecord
        p = re.compile('(.png)')
        filename = p.sub('.tfrecord', img_train_list[j])
        print(filename)
        with tf.python_io.TFRecordWriter(filename) as writer:
            writer.write(serialized_example)
        print(img_train_path, " elapse time: ",  time.time() - start_time)

img1_path = "/data/DIV2K/DIV2K_train_HR/0800.png"
img1 = load_image(img1_path)
print("successfully loaded img 1", img1.shape)


