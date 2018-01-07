#!/usr/bin/env python

import cv2
import numpy as np
import os
import sys
import tensorflow as tf
import tqdm

TARGET_WIDTH = 128
TARGET_HEIGH = 192

def get_src_file_paths(src_dir):
  for dirpath, _, filenames in os.walk(src_dir):
    for filename in filenames:
      if filename.endswith('.jpg'):
        yield os.path.join(dirpath, filename)


def get_image_data_from_file(file_path):
  image_data = cv2.imread(file_path)
  assert image_data.shape == (TARGET_HEIGH, TARGET_WIDTH, 3)
  # Swap  channnels in the way, preferred for TensorFlow
  image_data[:,:,0], image_data[:,:,2] = image_data[:,:,2], image_data[:,:,0]
  return image_data


def process(cropper_images_dir, dest_file_name):
  files_list = list(get_src_file_paths(cropper_images_dir))
  writer = tf.python_io.TFRecordWriter(dest_file_name)
  try:
    for file_name in tqdm.tqdm(files_list, unit='image'):
      image_data = get_image_data_from_file(file_name)
      image_bytes = image_data.astype(np.byte).tostring()
      example = tf.train.Example(features=tf.train.Features(feature={
        'image_bytes': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[image_bytes]))
      }))
      writer.write(example.SerializeToString())
  finally:
    writer.close()


def main():
  if len(sys.argv) != 3:
    sys.stderr.write('Converts images to tensorflow file\n');
    sys.stderr.write(
        'Usage: {} <cropper_out_dir> <dest file_name>\n'.format(sys.argv[0]))
    sys.exit(1)
  process(sys.argv[1], sys.argv[2])


if __name__ == '__main__':
  main()
