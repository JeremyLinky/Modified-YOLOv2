from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import utils.tfrecord_voc_utils as voc_utils

# annotations path, image path, save path, tfrecord prefix, shard
annotations_path = os.path.join(os.getcwd(), '..', 'voc2007', 'Annotation')
JpegImages_path = os.path.join(os.getcwd(), '..', 'voc2007', 'JPEGImage')
tfrecord = voc_utils.dataset2tfrecord(annotations_path, JpegImages_path,
                                      '../data1/', 'train', 1)
print(tfrecord)
