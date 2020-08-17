sudo apt-get install protobuf-compiler

virtualenv venv
source venv/bin/activate

pip install Cython
pip install contextlib2
pip install pillow
pip install lxml
pip install jupyter
pip install matplotlib
pip install pycocotools
pip install tqdm
pip install pandas

pip install tensorflow-gpu==1.15

export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

protoc object_detection/protos/*.proto --python_out=.

python object_detection/builders/model_builder_tf1_test.py

wget http://download.tensorflow.org/models/object_detection/mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz
tar -zxf mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz
