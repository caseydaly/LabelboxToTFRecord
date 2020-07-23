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


import json

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



#create a list of img_obj dictionaries from labelbox json format
def parse_labelbox_data(labelbox_data_path):
    f = open(labelbox_data_path)
    data = json.load(f)
    for record in data:
        label_objs = record["Label"]["objects"]
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df



def get_labels_from_json(labelbox_data_path):
    f = open(labelbox_data_path)
    data = json.load(f)
    labels_set = set()
    for record in data:
        label_objs = record["Label"]["objects"]
        for obj in label_objs:
            labels_set.add(obj["value"])
    print(labels_set)
    labels_list = list(labels_set)
    labels = {}
    for i in range(0, len(labels_list)):
        labels[labels_list[i]] = i
    return labels

    

print(parse_labelbox_json("../examples/example_1/labelbox_data.json"))