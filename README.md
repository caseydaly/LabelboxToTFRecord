# LabelboxToTFRecord
Convert Labelbox style json files to TFRecord file format (.tfrecord files) so the data can be used with TensorFlow.

## Installation

### Python Installation using Tensorflow 2:
You must have TensorFlow's Object Detection API installed, directions for installation can be found here: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md

Then run

`python3 -m pip install -r requirements.txt`

### Docker Installation using Tensorflow 1:

Assuming you have a config.yaml file set up (see Examples section):

```
# From project root
docker build -t lb2tf .
mkdir data

# This will run convert.py, downloading the data to a ./data folder
docker run --mount "type=bind,src=${PWD}/data,dst=/data" lb2tf --split 80 20 --download --labelbox-dest /data/labelbox --tfrecord-dest /data/tfrecord
```
Change the mount src to change where the data is downloaded to.

*NOTE:* if you have downloaded a large amount of data in your project, when `docker build` runs it will copy the data as part of the context which may take a long time. To avoid this, either move downloaded data outside of the project folder before doing a build, use mount settings to save the data outside the project folder to begin with, or use a dockerignore file to ignore the data once downloaded.

If you encounter permissions denied errors, check to see that docker hasn't created the `data` directory as root. `chown` or recreate the directory yourself to fix.

## Usage:

#### convert.py

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
      --limit LIMIT         Only retrieve and convert the first LIMIT data items
      --splits SPLITS [SPLITS ...]
                        Space-separated list of integer percentages for
                        splitting the output into multiple TFRecord files
                        instead of one. Sum of values should be <=100.
                        Example: '--splits 10 70' will write 3 files with 10%,
                        70%, and 20% of the data, respectively
      --download            Save the images locally in addition to creating
                        TFRecord(s)

#### split.py

    usage: split.py [-h] infile splits [splits ...]

    Split a .tfrecord file into smaller files

    positional arguments:
      infile      the .tfrecord file to split
      splits      Space-separated list of integers, the number of records to put
                  in each output file. Should add up to the total number of
                  records in the input tfrecord file.

    optional arguments:
      -h, --help  show this help message and exit


## Examples

Download Labelbox images and convert labels to TFRecord format:

`python convert.py PUID API_KEY`      ---     will download all Labelbox data (images, label file) to ./labelbox, will output tfrecord file to tfrecord/<PUID>.tfrecord
  
If you have a config.yaml file specified in the current directory, you can simply use...

`python convert.py`

To split data into two groups, with 30% in the first and 70% in the second...

`python convert.py --split 30 70`

To split data into two groups, with 30% in the first and 70% in the second, while downloading images locally...

`python convert.py --download --split 30 70`

You can also split an existing .tfrecord file into smaller pieces with split.py:

`python split.py ./10_record_file.tfrecord 3 2 5`

This will write 3 new files containing 3, 2, and 5 records, respectively.

# Tests

To run the tests:

```
cd src
python -m unittest
```
