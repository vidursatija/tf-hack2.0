from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import random
import sys
import threading
from PIL import Image
from tqdm import tqdm

import numpy as np
import six
import tensorflow as tf

VGG_MEAN = [103.94, 116.78, 123.68]

def synset_file_to_label(fname):
    lines = open(fname, 'r').readlines()
    synset_to_human = {}
    for i, l in enumerate(lines):
        if l:
            parts = l.strip().split(' ')
            synset = parts[0]
            human = " ".join(parts[1:])
            synset_to_human[synset] = (i, human)
    return synset_to_human

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def map_func(image_path, label, is_training):
    im = Image.open(image_path)
    numpy_img = np.array(im)
    h, w, c = numpy_img.shape
    # random crop and resize
    # subtract from mean and rgb to bgr
    
    if is_training:
        # For training, we want to randomize some of the distortions.
        if h < w:
            crop_dim = random.randint(0, w-h)
            numpy_img = numpy_img[:, crop_dim: crop_dim+h, :]
            im2 = Image.fromarray(numpy_img)
            im = np.array(im2.resize((224, 224), Image.BILINEAR))
        else:
            crop_dim = random.randint(0, h-w)
            numpy_img = numpy_img[crop_dim: crop_dim+w, :, :]
            im2 = Image.fromarray(numpy_img)
            im = np.array(im2.resize((224, 224), Image.BILINEAR))
    else:
        # For validation, we want to decode, resize, then just crop the middle.
        resize_min = 256
        smaller_dim = min(h, w)
        scale_ratio = resize_min / smaller_dim
        n_h, n_w = int(h*scale_ratio), int(w*scale_ratio)
        im = im.resize((n_w, n_h), Image.BILINEAR)
        numpy_img = np.array(im)
        
        amount_to_be_cropped_h = (h - 224)
        crop_top = amount_to_be_cropped_h // 2
        crop_bottom = (amount_to_be_cropped_h+1) // 2
        amount_to_be_cropped_w = (w - 224)
        crop_left = amount_to_be_cropped_w // 2
        crop_right = (amount_to_be_cropped_w+1) // 2
        im = numpy_img[crop_top: -crop_bottom, crop_left: -crop_right, :]
        
    r = im[:, :, 0] - VGG_MEAN[0]
    g = im[:, :, 1] - VGG_MEAN[1]
    b = im[:, :, 2] - VGG_MEAN[2]
    im2 = np.stack([b, g, r], axis=-1)

    feature = {'image': _float_feature(im2.reshape([-1]).tolist()),
        'label': _int64_feature([label])}
    example = tf.train.Example(features=tf.train.Features(feature=feature))

    return example

train_path = sys.argv[1] # all image files in the form x_y.JPEG where x is synset id
val_path = sys.argv[2] # all image files
val_text_file = sys.argv[3] # all validation ground truth file
synset_file = sys.argv[4] # synset file

examples_per_file = 3200

train_files = [f for f in os.listdir(train_path) if f.endswith("JPEG") or f.endswith("jpeg")]
val_files = [os.path.join(val_path, f) for f in os.listdir(val_path) if f.endswith("JPEG") or f.endswith("jpeg")] # [:2500]
val_files.sort()
val_lines = open(val_text_file, "r").readlines()
synset_dict = synset_file_to_label(synset_file)

print(len(train_files))
print(len(val_files))
print(len(val_lines))

train_data_files = (len(train_files)+examples_per_file)//examples_per_file

output_train_path = "tfdatasets/train"
output_test_path = "tfdatasets/test"

for i in tqdm(range(train_data_files)):
    train_filename = 'train-'+str(i)+'.tfrecords'  # address to save the TFRecords file
    writer = tf.io.TFRecordWriter(os.path.join(output_train_path, train_filename))
    train_batch = train_files[i*examples_per_file: (i+1)*examples_per_file]
    for f in tqdm(train_batch):
        try:
            ss, _ = f.split("_")
            label = synset_dict[ss][0]
            ff = os.path.join(train_path, f)
            ex = map_func(ff, label, True)
            writer.write(ex.SerializeToString())
        except Exception as e:
            print(e)
    writer.close()

test_filename = 'test-0.tfrecords'  # address to save the TFRecords file
writer = tf.io.TFRecordWriter(os.path.join(output_test_path, test_filename))
for i, f in tqdm(enumerate(val_files)):
    try:
        label = int(val_lines[i])-1
        ex = map_func(f, label, False)
        writer.write(ex.SerializeToString())
    except Exception as e:
            print(e)
writer.close()