from labelbox import Client
import json
import urllib.request
from urllib.parse import urlparse
import io
from PIL import Image
import PIL
import requests
import os
from os import path
import tensorflow as tf
import label
import time
import progressbar

#contains all of the necessary info to create a tfrecord
class TFRecordInfo:

    def __init__(self, height, width, filename, source_id, encoded, format, labels):
        self.height = height
        self.width = width
        self.filename = filename
        self.source_id = source_id
        self.encoded = encoded
        self.format = format
        self.labels = labels

    def __repr__(self):
        return "TFRecordInfo({0}, {1}, {2}, {3}, {4}, {5}, {6})".format(self.height, self.width, self.filename, self.source_id, type(self.encoded), self.format, self.labels)

#download data and create a list of img_obj dictionaries from labelbox json format
def parse_labelbox_data(project_unique_id, api_key, labelbox_dest, download):
    data = retrieve_data(project_unique_id, api_key, labelbox_dest)
    image_format = b'jpg'
    records = list()
    print("Retrieving images from Labelbox...")
    if download:
        print("--download flag detected")
        print("Downloading images locally to labelbox/images")
    bar = progressbar.ProgressBar(maxval=len(data), \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    for i in range(len(data)):
        record = data[i]
        jpg_url = record["Labeled Data"]
        temp = urlparse(jpg_url)
        image_name = temp.path
        image_name = image_name[1:]
        if not os.path.exists(labelbox_dest):
            os.makedirs(labelbox_dest)
        if labelbox_dest[len(labelbox_dest)-1] != '/':
            labelbox_dest += '/'

        if download:
            if not os.path.exists(labelbox_dest+"images"):
                os.makedirs(labelbox_dest+"images")
            outpath = labelbox_dest+"images/"+image_name
            if not path.exists(outpath):
                jpg = urllib.request.urlretrieve(jpg_url, outpath)
            with tf.io.gfile.GFile(outpath, 'rb') as fid:
                encoded_jpg = fid.read()
                im = Image.open(outpath)
                width, height = im.size
        else:
            with urllib.request.urlopen(jpg_url) as url:
                encoded_jpg = io.BytesIO(url.read()).read()
                im = Image.open(io.BytesIO(encoded_jpg))
                width, height = im.size
        labels = list()
        if "objects" in record["Label"]:
            label_objs = record["Label"]["objects"]
            for l in label_objs:
                labels.append(label.label_from_labelbox_obj(l))
            records.append(TFRecordInfo(height, width, outpath, outpath, encoded_jpg, image_format, labels))
        bar.update(i+1)
    bar.finish()
    return data, records

# def image_to_byte_array(image:Image):
#   imgByteArr = io.BytesIO()
#   image.save(imgByteArr, format=image.format)
#   imgByteArr = imgByteArr.getvalue()
#   return imgByteArr

def get_classes_from_labelbox(data):
    labels_set = set()
    for record in data:
        if isinstance(record, dict):
            if "objects" in record["Label"]:
                label_objs = record["Label"]["objects"]
                for obj in label_objs:
                    labels_set.add(obj["value"])
    labels_list = list(labels_set)
    labels = {}
    for i in range(0, len(labels_list)):
        labels[labels_list[i]] = i
    return labels

#retrieve data from labelbox
def retrieve_data(project_unique_id, api_key, labelbox_dest):
    client = Client(api_key)
    project = client.get_project(project_unique_id)
    retrieve_url = project.export_labels()
    with urllib.request.urlopen(retrieve_url) as url:
        response = url.read()
        data = json.loads(response)
    if not os.path.exists(labelbox_dest):
        os.makedirs(labelbox_dest)
    if labelbox_dest[len(labelbox_dest)-1] != '/':
        labelbox_dest += '/'
    outpath = labelbox_dest + project_unique_id + ".json"
    with open(outpath, 'w') as outfile:
        json.dump(data, outfile)
    return data
