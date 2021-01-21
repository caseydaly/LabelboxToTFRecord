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
import hashlib
import re

#contains all of the necessary info to create a tfrecord
class TFRecordInfo:

    def __init__(self, height, width, filename, source_id, encoded, format, sha_key, labelbox_rowid, labelbox_url, labels):
        self.height = height
        self.width = width
        self.filename = filename
        self.source_id = source_id
        self.encoded = encoded
        self.format = format
        self.sha_key = sha_key
        self.labelbox_rowid = labelbox_rowid
        self.labelbox_url = labelbox_url
        self.labels = labels

    def __repr__(self):
        return "TFRecordInfo({0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8})".format(self.height, self.width, self.filename, self.source_id, type(self.encoded), self.format, self.sha_key, self.labelbox_rowid, self.labels)

#download data and create a list of img_obj dictionaries from labelbox json format
def parse_labelbox_data(project_unique_id, api_key, labelbox_dest, download, limit):
    data = retrieve_data(project_unique_id, api_key, labelbox_dest)
    image_format = b'jpg'
    records = list()
    print("Retrieving images from Labelbox...")
    if download:
        print("--download flag detected")
        print("Downloading images locally to labelbox/images")
    if len(data) > limit:
        print ("--limit flag detected")
        print(f"Found {len(data)} items but limiting to {limit}")
    bar = progressbar.ProgressBar(maxval=min(len(data), limit), \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    skipped = 0
    for i in range(min(len(data), limit)):
        record = data[i]

        # TODO Maybe we want some images without labels
        if "objects" in record["Label"]:

            jpg_url = record["Labeled Data"]
            temp = urlparse(jpg_url)
            # This seems to be the org id + guid (image specific for storage purposes?) + external id (which is the original filename)
            #image_name = temp.path[1:]
            # TODO what if external id isn't defined...
            # DataRow ID seems to be what is used on the Labels tab anyway...
            image_name = f"{record['ID']}-{record['DataRow ID']}-{record['External ID']}"
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
                # TODO use record["External ID"], which appears to be filename?
                outpath = ""
                with urllib.request.urlopen(jpg_url) as url:
                    # TODO Not sure if this matters. I think this maybe could be one less read, like:
                    # encoded_jpg = url.read()
                    # im = Image.open(io.BytesIO(encoded_jpg))
                    encoded_jpg = io.BytesIO(url.read()).read()
                    im = Image.open(io.BytesIO(encoded_jpg))
                    width, height = im.size

            sha_key = hashlib.sha256(encoded_jpg).hexdigest()

            labels = list()
            label_objs = record["Label"]["objects"]
            for l in label_objs:
                labels.append(label.label_from_labelbox_obj(l))

            source_id = re.sub(r'\.(jpe?g|png)$', '', f"{record['DataRow ID']}-{record['External ID']}")
            source_id = re.sub(r'\.','_', source_id)
            records.append(TFRecordInfo(height, width, record["External ID"], source_id, encoded_jpg, image_format, sha_key, record["DataRow ID"], record["View Label"], labels))
        else:
            skipped += 1
            print(f"DataRow {record['DataRow ID']} has no labels. Skipping. See more at {record['View Label']}\n")
        bar.update(i+1)
    bar.finish()

    print(f"Found {len(data)} images:")
    print(f"  Saved: {len(records)}")
    print(f"  Skipped: {skipped}")

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
