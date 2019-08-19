from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from lxml.html import etree
import os
import numpy as np
import warnings
import math
import sys
from utils.voc_classname_encoder import classname_to_ids
from utils.image_augmentor import image_augmentor

def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(bytes_list=tf.train.Int64List(value=values))


def bytes_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def float_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(bytes_list=tf.train.FloatList(value=values))


def xml_to_example(xmlpath, imgpath):
    xml = etree.parse(xmlpath)
    root = xml.getroot()
    imgname = root.find('filename').text
    imgname = os.path.join(imgpath, imgname)
    image = tf.gfile.GFile(imgname, 'rb').read()
    size = root.find('size')
    height = int(size.find('height').text)
    width = int(size.find('width').text)
    depth = int(size.find('depth').text)
    shape = np.asarray([height, width, depth], np.int32)
    xpath = xml.xpath('//object')
    ground_truth = np.zeros([len(xpath), 5], np.float32)
    for i in range(len(xpath)):
        obj = xpath[i]
        classid = classname_to_ids[obj.find('name').text]
        bndbox = obj.find('bndbox')
        ymin = float(bndbox.find('ymin').text)
        ymax = float(bndbox.find('ymax').text)
        xmin = float(bndbox.find('xmin').text)
        xmax = float(bndbox.find('xmax').text)
        ground_truth[i, :] = np.asarray([ymin, ymax, xmin, xmax, classid], np.float32)      #GT所包含的信息
    features = {
        'image': bytes_feature(image),
        'shape': bytes_feature(shape.tobytes()),
        'ground_truth': bytes_feature(ground_truth.tobytes())
    }
    example = tf.train.Example(features=tf.train.Features(
        feature=features))
    return example


def dataset2tfrecord(xml_dir, img_dir, output_dir, name, total_shards=5):
    if not tf.gfile.Exists(output_dir):
        tf.gfile.MakeDirs(output_dir)
        print(output_dir, 'does not exist, create it done')
    else:
        if len(tf.gfile.ListDirectory(output_dir)) == 0:
            print(output_dir, 'already exist, need not create new')
        else:
            warnings.warn(output_dir + ' is not empty!', UserWarning)
    outputfiles = []
    xmllist = tf.gfile.Glob(os.path.join(xml_dir, '*.xml'))
    num_per_shard = int(math.ceil(len(xmllist)) / float(total_shards))
    for shard_id in range(total_shards):
        outputname = '%s_%05d-of-%05d.tfrecord' % (name, shard_id+1, total_shards)
        outputname = os.path.join(output_dir, outputname)
        outputfiles.append(outputname)
        with tf.python_io.TFRecordWriter(outputname) as tf_writer:
            start_ndx = shard_id * num_per_shard
            end_ndx = min((shard_id+1) * num_per_shard, len(xmllist))
            for i in range(start_ndx, end_ndx):
                sys.stdout.write('\r>> Converting image %d/%d shard %d/%d' % (
                    i+1, len(xmllist), shard_id+1, total_shards))
                sys.stdout.flush()
                example = xml_to_example(xmllist[i], img_dir)
                tf_writer.write(example.SerializeToString())
            sys.stdout.write('\n')
            sys.stdout.flush()
    return outputfiles


class Dataset:
    def parse_function(self, data, config):
            features = tf.io.parse_single_example(data, features={     # 产生一个实例
                'image': tf.io.FixedLenFeature([], tf.string),
                'shape': tf.io.FixedLenFeature([], tf.string),
                'ground_truth': tf.io.FixedLenFeature([], tf.string)
            })
            shape = tf.decode_raw(features['shape'], tf.int32)      # tf.decode_raw:主要用于将原来编码为字符串的类型变回来
            ground_truth = tf.decode_raw(features['ground_truth'], tf.float32)
            shape = tf.reshape(shape, [3])
            ground_truth = tf.reshape(ground_truth, [-1, 5])    # 这个框的标注对应于5个内容
            images = tf.image.decode_jpeg(features['image'], channels=3)    # channel=3为RGB图片
            images = tf.cast(tf.reshape(images, shape), tf.float32)
            images, ground_truth = image_augmentor(image=images,
                                                   input_shape=shape,
                                                   ground_truth=ground_truth,
                                                   **config
                                                   )
            return images, ground_truth

    def get_generator(self, tfrecords, batch_size, buffer_size, image_preprocess_config):     # image_preprocess_config:数据增强的config
        self.data = tf.data.TFRecordDataset(tfrecords)       # 读取tfrecord格式的数据集
        # data1 = data.map(lambda x: parse_function(x, image_preprocess_config))

        self.data = self.data.map(lambda x: self.parse_function(x, image_preprocess_config)).shuffle(buffer_size=buffer_size).batch(batch_size, drop_remainder=True).repeat()

        self.iterator = tf.compat.v1.data.Iterator.from_structure(tf.compat.v1.data.get_output_types(self.data),
                                                                  tf.compat.v1.data.get_output_shapes(self.data))
        # This iterator-constructing method can be used to create an iterator that
        # is reusable with many different datasets

        self.init_op = self.iterator.make_initializer(self.data)
        # A `tf.Operation` that can be run to initialize this iterator on the given `dataset`
        return self.init_op, self.iterator

