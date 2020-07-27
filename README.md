# LabelboxToTFRecord
Download Labelbox data, convert to TFRecord, and store a local TFRecord file. Optionally, just do a download of Labelbox images.

You must have TensorFlow's Object Detection API installed, directions for installation can be found here: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md

# Usage:

usage: convert.py [-h] [--labelbox-dest LABELBOX_DEST] [--tfrecord-dest TFRECORD_DEST] [--download-only] PUID API_KEY

Download Labelbox data, convert to TFRecord, and store a local TFRecord file. Optionally, just do a download of Labelbox images.

positional arguments:
  PUID                  Project Unique ID (PUID) of your Labelbox project,
                        found in URL of Labelbox project home page
  API_KEY               API key associated with your Labelbox account

optional arguments:
  -h, --help            show this help message and exit
  --labelbox-dest LABELBOX_DEST
                        Destination folder for downloaded images and json file
                        of Labelbox labels.
  --tfrecord-dest TFRECORD_DEST
                        Destination folder for downloaded images
  --download-only       Use this flag if you only want to download the images
                        and not convert to TFRecord format.

# Examples

Download Labelbox images and convert labels to TFRecord format:

python convert.py PUID API_KEY      ---     will download all Labelbox data (images, label file) to ./labelbox, will output tfrecord file to tfrecord/<PUID>.tfrecord

Just download Labelbox images:

python convert.py PUID API_KEY --download-only      --- will download all Labelbox data (images, label file) to ./labelbox


