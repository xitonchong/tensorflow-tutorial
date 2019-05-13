import tensorflow as tf
import numpy as np
import scipy.misc
import glob

tf.enable_eager_execution()

#img1_dir = '/data/DIV2K/DIV2K_train_LR_unknown/X4/0324x4.tfrecord'
img_dir = glob.glob('/data/DIV2K/DIV2K_train_LR_unknown/X4/*.tfrecord')
raw_image_dataset = tf.data.TFRecordDataset(img_dir)

img_feature_description = {
        'train': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.string)
}

def _parse_img_fn(example_proto):
    return tf.parse_single_example(example_proto, img_feature_description)

parsed_img_dataset = raw_image_dataset.map(_parse_img_fn)
print("==", parsed_img_dataset)

for img_features in parsed_img_dataset:
    train = img_features['train'].numpy()
    train = tf.io.decode_image(train)
    label = img_features['label'].numpy()
    label = tf.io.decode_image(label)
    print("img type ", type(train))
    print("img label shape ", label.shape)
    print("img train shape: ", train.shape)
#scipy.misc.imsave("train1.png", train)
#scipy.misc.imsave("label1.png", label)
