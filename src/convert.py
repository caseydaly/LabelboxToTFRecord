from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import sys
import pandas as pd
import tensorflow as tf
import glob
import xml.etree.ElementTree as ET
import parse_labelbox
import argparse
import yaml
from pathlib import Path
from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict


def create_tf_example(record_obj, class_dict):

    filename = record_obj.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for label_obj in record_obj.labels:
        xmins.append(label_obj.xmin / record_obj.width)
        xmaxs.append(label_obj.xmax / record_obj.width)
        ymins.append(label_obj.ymin / record_obj.height)
        ymaxs.append(label_obj.ymax / record_obj.height)
        classes_text.append(label_obj.label.encode('utf8'))
        classes.append(class_dict[label_obj.label])

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(record_obj.height),
        'image/width': dataset_util.int64_feature(record_obj.width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(record_obj.encoded),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def generate_records(puid, api_key, labelbox_dest, tfrecord_dest):
    data, records = parse_labelbox.parse_labelbox_data(puid, api_key, labelbox_dest)
    class_dict = parse_labelbox.get_classes_from_labelbox(data)
    tfrecord_folder = tfrecord_dest
    if not os.path.exists(tfrecord_folder):
        os.makedirs(tfrecord_folder)
    if tfrecord_folder[len(tfrecord_folder)-1] != '/':
        tfrecord_folder += '/'
    outfile = puid + ".tfrecord"
    outpath = tfrecord_folder + outfile
    with tf.io.TFRecordWriter(outpath) as writer:
        for record in records:
            tf_example = create_tf_example(record, class_dict)
            writer.write(tf_example.SerializeToString())
    print('Successfully created the TFRecords at location: {}'.format(outpath))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download Labelbox data, and optionally convert to TFRecord format.')
    parser.add_argument('--puid', help="Project Unique ID (PUID) of your Labelbox project, found in URL of Labelbox project home page")
    parser.add_argument('--api-key', help="API key associated with your Labelbox account")
    parser.add_argument('--labelbox-dest', help="Destination folder for downloaded images and json file of Labelbox labels.", default="labelbox")
    parser.add_argument('--tfrecord-dest', help="Destination folder for downloaded images", default="tfrecord")
    parser.add_argument('--download-only', help="Use this flag if you only want to download the images and not convert to TFRecord format.", action='store_true')
    args = parser.parse_args()

    config_path = (Path(os.path.realpath(__file__)).parent / Path("config.yaml")).resolve()
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    puid = args.puid or config['puid']
    api_key = args.api_key or config['api_key']

    if args.download_only:
        parse_labelbox.parse_labelbox_data(puid, api_key, args.labelbox_dest)
    else:
        generate_records(puid, api_key, args.labelbox_dest, args.tfrecord_dest)

    


