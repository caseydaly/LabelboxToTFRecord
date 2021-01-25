FROM tensorflow/tensorflow:1.15.2-py3
# To switch to TensorFlow 2, change the above to a TensorFlow 2 image.
# You will also have to change the tf1/setup.py reference below to use tf2/setup.py

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    protobuf-compiler

RUN python -m pip install -U pip
RUN python -m pip install mypy

# Currently using xdhmoore's repo to ensure compatibility for xdhmoore's stuff. Uncomment to use the main repo instead,
# Though it's probably advisable to pin to a specific commit, not master (since master changes often)
# RUN git clone --branch master --single-branch https://github.com/tensorflow/models.git /tfmodels
RUN git clone --branch lb2tf --single-branch https://github.com/xdhmoore/models.git /tfmodels

RUN cp /tfmodels/research/object_detection/packages/tf1/setup.py /tfmodels/research/
RUN cd /tfmodels/research/ && protoc object_detection/protos/*.proto --python_out=.
RUN cd /tfmodels/research && python -m pip install .

# Add new user to avoid running as root
RUN useradd -ms /bin/bash labelboxtotfrecord

USER labelboxtotfrecord
WORKDIR /home/labelboxtotfrecord/LabelboxToTFRecord

COPY requirements.txt requirements.txt
RUN python -m pip install -r requirements.txt


# TODO fix this so it doesn't copy api key?
# The python code could also be changed to use environment variables
COPY src /home/labelboxtotfrecord/LabelboxToTFRecord/src

VOLUME ["/data"]

# TODO tweak this to make it possible to call split.py or shuffle.py
ENTRYPOINT [ "python3", "/home/labelboxtotfrecord/LabelboxToTFRecord/src/convert.py" ]
CMD [ "--help" ]
