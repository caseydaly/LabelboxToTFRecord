# LabelboxToTFRecord
Convert Labelbox style json files to TFRecord file format (.tfrecord files) so the data can be used with TensorFlow.

You must have TensorFlow's Object Detection API installed, directions for installation can be found here: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md

## Usage:

    usage: convert.py [-h] [--puid PUID] [--api-key API_KEY]
                  [--labelbox-dest LABELBOX_DEST]
                  [--tfrecord-dest TFRECORD_DEST]
                  [--splits SPLITS [SPLITS ...]] [--download]

    Convert Labelbox data to TFRecord and store .tfrecord file(s) locally. Saving
    images to disk is optional. Create a file "config.yaml" in the current
    directory to store Labelbox sensitive data, see "config.yaml.sample" for an
    example.

    optional arguments:
      -h, --help            show this help message and exit
      --puid PUID           Project Unique ID (PUID) of your Labelbox project,
                        found in URL of Labelbox project home page
      --api-key API_KEY     API key associated with your Labelbox account
      --labelbox-dest LABELBOX_DEST
                        Destination folder for downloaded images and json file
                        of Labelbox labels.
      --tfrecord-dest TFRECORD_DEST
                        Destination folder for downloaded images
      --splits SPLITS [SPLITS ...]
                        Space-separated list of integer percentages for
                        splitting the output into multiple TFRecord files
                        instead of one. Sum of values should be <=100.
                        Example: '--splits 10 70' will write 3 files with 10%,
                        70%, and 20% of the data, respectively
      --download            Save the images locally in addition to creating
                        TFRecord(s)

## Examples

Download Labelbox images and convert labels to TFRecord format:

`python convert.py PUID API_KEY`      ---     will download all Labelbox data (images, label file) to ./labelbox, will output tfrecord file to tfrecord/<PUID>.tfrecord
  
If you have a config.yaml file specified in the current directory, you can simply use...

`python convert.py`

To split data into two groups, with 30% in the first and 70% in the second...

`python convert.py --split 30 70`

To split data into two groups, with 30% in the first and 70% in the second, while downloading images locally...

`python convert.py --download --split 30 70`



