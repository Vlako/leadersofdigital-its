import tensorflow as tf
import pandas as pd
from tqdm import tqdm
import json

from object_detection.utils import dataset_util
from PIL import Image

import contextlib2
from object_detection.dataset_tools import tf_record_creation_util

flags = tf.app.flags
flags.DEFINE_string('subset', 'train', 'Path to output TFRecord')
FLAGS = flags.FLAGS

print(FLAGS.subset)

subset = FLAGS.subset


def create_tf_example(example):
  # TODO(user): Populate the following variables from your example.
  img = Image.open(f'../train/train/'+example['ImageID']+'.png')

  height = img.height # Image height
  width = img.width # Image width
  filename = example['ImageID'].encode() + b'.jpg' # Filename of the image. Empty if image is not from file

  with open(f'../train/train/'+example['ImageID'] + '.png', 'rb') as image_file:
    encoded_image_data = image_file.read()
#  encoded_image_data = img.tobytes() # Encoded image bytes
  image_format = b'png' # b'jpeg' or b'png'

  xmins = example['XMin'] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = example['XMax'] # List of normalized right x coordinates in bounding box
             # (1 per box)
  ymins = example['YMin'] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = example['YMax'] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
  classes_text = [i.encode() for i in example['Label']] # List of string class name of bounding box (1 per box)
  classes = example['LabelName'] # List of integer class id of bounding box (1 per box)
  
  masks = [
      open(f'../mask/'+i + '.png', 'rb').read()
      for i in example['MaskName']
  ]

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/mask': dataset_util.bytes_list_feature(masks)
  }))
  return tf_example



def main(_):
    # TODO(user): Write code to read in your dataset to examples variable
    meta = pd.read_csv(f'../{subset}.csv', dtype={'ImageID': str})
    meta = meta.groupby('ImageID').agg({
        'LabelName': list, 
        'Label': list, 
        'XMax': list, 
        'XMin': list, 
        'YMax': list, 
        'YMin': list,
        'MaskName': list
    }).reset_index()

    num_shards = 1
    output_filebase = f'data/{subset}'

    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
            tf_record_close_stack, output_filebase, num_shards)
        for example in tqdm(meta.T.to_dict().values()):
            tf_example = create_tf_example(example)
            index = hash(example['ImageID'])
            output_shard_index = index % num_shards
            output_tfrecords[output_shard_index].write(tf_example.SerializeToString())


if __name__ == '__main__':
    tf.app.run()
