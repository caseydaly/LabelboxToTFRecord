"""
img_obj is of format
{
    'filename': filename,
    'width': width,
    'height': height,
    'class': class,
    'labels': labels --- do we need this? 
    'xmin': xmin, 
    'ymin': ymin, 
    'xmax': xmax, 
    'ymax': ymax
}
"""

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
import api_info

# example image url: https://m.media-amazon.com/images/S/aplus-media/vc/6a9569ab-cb8e-46d9-8aea-a7022e58c74a.jpg
def download_image(url, image_file_path):
    r = requests.get(url, timeout=4.0)
    if r.status_code != requests.codes.ok: #pylint: disable=no-member
        assert False, 'Status code error: {}.'.format(r.status_code)

    with Image.open(io.BytesIO(r.content)) as im:
        #im.save(image_file_path)
        print(type(im))

    print('Image downloaded from url: {} and saved to: {}.'.format(url, image_file_path))

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

#create a list of img_obj dictionaries from labelbox json format
def parse_labelbox_data(project_unique_id, api_key, local_folder):
    data = retrieve_data(project_unique_id, api_key)
    image_format = b'jpg'
    records = list()
    for record in data:
        jpg_url = record["Labeled Data"]
        temp = urlparse(jpg_url)
        image_name = temp.path
        image_name = image_name[1:]
        if not os.path.exists(local_folder):
            os.makedirs(local_folder)
        if local_folder[len(local_folder)-1] != '/':
            local_folder += '/'
        outpath = local_folder+image_name
        if not path.exists(outpath):
            jpg = urllib.request.urlretrieve(jpg_url, local_folder+image_name)
        with tf.io.gfile.GFile(outpath, 'rb') as fid:
            encoded_jpg = fid.read()
        im = Image.open(outpath)
        width, height = im.size
        labels = list()
        for x in record["Label"]:
            print(x)
        if "objects" in record["Label"]:
            label_objs = record["Label"]["objects"]
            for l in label_objs:
                labels.append(label.label_from_labelbox_obj(l))
            records.append(TFRecordInfo(height, width, outpath, outpath, encoded_jpg, image_format, labels))
    return records
    #column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    #xml_df = pd.DataFrame(xml_list, columns=column_name)
    #return xml_df



def get_labels_from_json(labelbox_data_path):
    f = open(labelbox_data_path)
    data = json.load(f)
    labels_set = set()
    for record in data:
        label_objs = record["Label"]["objects"]
        for obj in label_objs:
            labels_set.add(obj["value"])
    #print(labels_set)
    labels_list = list(labels_set)
    labels = {}
    for i in range(0, len(labels_list)):
        labels[labels_list[i]] = i
    return labels

#retrieve data from labelbox
def retrieve_data(project_unique_id, api_key):
    client = Client(api_key)
    project = client.get_project(project_unique_id)
    retrieve_url = project.export_labels()
    with urllib.request.urlopen(retrieve_url) as url:
        response = url.read()
        #  I'm guessing this would output the html source code ?
        data = json.loads(response)
        #print(data)
    return data