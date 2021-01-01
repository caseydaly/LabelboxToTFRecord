# vim:ts=4:sw=4:sts=4
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import sys
import tensorflow as tf
import glob
import xml.etree.ElementTree as ET
import parse_labelbox
import argparse
import yaml
import random
import progressbar
from datetime import datetime
from pathlib import Path
from PIL import Image

from object_detection.utils import dataset_util
from object_detection.protos import string_int_label_map_pb2
from google.protobuf import text_format

from collections import namedtuple, OrderedDict

# Maps a { 'shark': 1, 'person': 2 } dict to a labelmap structure
# for the pbtxt file
def class_dict_to_label_map_str(class_dict):
    label_map_proto = string_int_label_map_pb2.StringIntLabelMap()
    for key,val in class_dict.items():
        item = label_map_proto.item.add()
        item.name = key
        # 0 is reserved for 'background' only, which we aren't using
        item.id = val + 1

    return text_format.MessageToString(label_map_proto)

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
        'image/key/sha256': dataset_util.bytes_feature(record_obj.sha_key.encode('utf8')),
        'image/labelbox/datarow_id': dataset_util.bytes_feature(record_obj.labelbox_rowid.encode('utf8')),
        'images/labelbox/view_url': dataset_util.bytes_feature(record_obj.labelbox_url.encode('utf8')),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

# Convert a list of percentages adding to 100, ie [20, 30, 50] to a list of indices into the list of records
# at which to split off a new file
def splits_to_record_indices(splits, num_records):
    if splits is None or splits == []:
        splits = [100]
    if sum(splits) != 100: raise ValueError("Percentages must add to 100")
    if not splits or splits == [100]: return [num_records]

    prev_idx = 0
    img_indices = []
    for split_idx,split in enumerate(splits[:-1]):
        end_img_idx = round(prev_idx + (split / 100) * num_records)

        if end_img_idx == prev_idx:
            #Leave in dupes for now. Take out at the end
            pass
        else:
            img_indices += [min(end_img_idx, num_records)]
        prev_idx = end_img_idx

    # Adding the last index this way ensures that it isn't rounded down
    img_indices += [num_records]
    # Dedupe
    return list(OrderedDict.fromkeys(img_indices))

def generate_records(puid, api_key, labelbox_dest, tfrecord_dest, splits, download, limit):
    data, records = parse_labelbox.parse_labelbox_data(puid, api_key, labelbox_dest, download, limit)
    print("Creating .tfrecord files")
    class_dict = parse_labelbox.get_classes_from_labelbox(data)
    tfrecord_folder = tfrecord_dest
    if not os.path.exists(tfrecord_folder):
        os.makedirs(tfrecord_folder)
    if tfrecord_folder[len(tfrecord_folder)-1] != '/':
        tfrecord_folder += '/'

    strnow = datetime.now().strftime('%Y-%m-%dT%H%M')
    splits = splits_to_record_indices(splits, len(records))
    assert splits[-1] == len(records), f'{splits}, {len(records)}'

    # TODO use a seed-based random so it's the same every time
    random.shuffle(records)

    split_start = 0
    print(f'Creating {len(splits)} TFRecord files:')
    for split_end in splits:
        outfile = f'{puid}_{strnow}_{split_end - split_start}.tfrecord'
        outpath = tfrecord_folder + outfile
        with tf.io.TFRecordWriter(outpath) as writer:
            for record in records[split_start:split_end]:
                tf_example = create_tf_example(record, class_dict)
                writer.write(tf_example.SerializeToString())
        print('Successfully created TFRecord file at location: {}'.format(outpath))
        split_start = split_end

    pb_file_name = f'{puid}_{strnow}.pbtxt'
    label_text = class_dict_to_label_map_str(class_dict)
    with open(tfrecord_folder + pb_file_name, 'w') as label_file:
        print(f'Creating label map file')
        label_file.write(label_text)


def validate_splits(args_splits):
    if not args_splits:
        splits = [100]
    else:
        if any(s <= 0 for s in args_splits):
            parser.error(message='splits must be positive integers')

        if sum(args_splits) < 100:
            splits = args_splits + [100 - sum(args_splits)]
        elif sum(args_splits) > 100:
            parser.error("splits must sum up to <= 100")
        else:
            splits = args_splits

    splits.sort()

    return splits


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Labelbox data to TFRecord and store .tfrecord file(s) locally. Saving images to disk is optional. Create a file "config.yaml" in the current directory to store Labelbox sensitive data, see "config.yaml.sample" for an example.')
    parser.add_argument('--puid', help="Project Unique ID (PUID) of your Labelbox project, found in URL of Labelbox project home page")
    parser.add_argument('--api-key', help="API key associated with your Labelbox account")
    parser.add_argument('--labelbox-dest', help="Destination folder for downloaded images and json file of Labelbox labels.", default="labelbox")
    parser.add_argument('--tfrecord-dest', help="Destination folder for .tfrecord file(s)", default="tfrecord")
    parser.add_argument('--limit', help="Only retrieve and convert the first LIMIT data items", type=int, default=2**1000)
    parser.add_argument('--splits', help="Space-separated list of integer percentages for splitting the " +
        "output into multiple TFRecord files instead of one. Sum of values should be <=100. " +
        "Example: '--splits 10 70' will write 3 files with 10%%, 70%%, and 20%% of the data, respectively",
        nargs='+',
        type=int,
    )
    parser.add_argument('--download', help="Save the images locally in addition to creating TFRecord(s)", action='store_true')
    args = parser.parse_args()

    config_path = (Path(os.path.realpath(__file__)).parent / Path("config.yaml")).resolve()
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    puid = args.puid or config['puid']
    api_key = args.api_key or config['api_key']
    splits = validate_splits(args.splits)

    generate_records(puid, api_key, args.labelbox_dest, args.tfrecord_dest, splits, args.download, args.limit)

    


