#!/usr/bin/env python
import argparse
import sys
import subprocess
import tensorflow as tf
import tqdm

def make_writer(dst_file_name, width, height):
  size_str = '{}x{}'.format(width, height)
  command = [ 'ffmpeg',
            '-y', # (optional) overwrite output file if it exists
            '-f', 'rawvideo',
            '-vcodec','rawvideo',
            '-s',
            size_str, # size of one frame
            '-pix_fmt', 'rgb24',
            '-r', '24', # frames per second
            '-i', '-', # The imput comes from a pipe
            '-an', # Tells FFMPEG not to expect any audio
            '-vcodec', 'mpeg4',
            dst_file_name ]
  print ' '.join(command)
  return subprocess.Popen(command, stdin=subprocess.PIPE)

def parse_args():
  parser = argparse.ArgumentParser('Trains GAN neural network')
  parser.add_argument(
      'src_file_name',
      help='File created by tf.summary.FileWriter')
  parser.add_argument(
      'dst_file_name',
      help='Destination MP4 file name')
  parser.add_argument(
      'image_tag',
      help='Tag of the image, from which make the video')
  parser.add_argument(
      '--frame-count',
      action='store',
      help='Limit number processed images')
  return parser.parse_args()

def main():
  options = parse_args()
  frame_count = int(options.frame_count) if options.frame_count is not None else None
  image_str = tf.placeholder(tf.string)
  im_tf = tf.image.decode_image(image_str)
  writer = None
  with tf.Session() as session:
    with session.as_default():
      records_done = 0
      done = False
      try:
        for e in tf.train.summary_iterator(options.src_file_name):
          if done:
            break
          for v in e.summary.value:
            if v.tag == options.image_tag:
              im = im_tf.eval({image_str: v.image.encoded_image_string})
              if not writer:
                writer = make_writer(options.dst_file_name, im.shape[1], im.shape[0])
              writer.stdin.write(im.tobytes())
              records_done += 1
              if frame_count is not None and records_done > frame_count:
                done = True
                break
      finally:
        print '{} records done'.format(records_done)

if __name__ == '__main__':
  main()
