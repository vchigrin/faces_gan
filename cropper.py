#!/usr/bin/env python

import cv2
import os
import sys
import tqdm

TARGET_WIDTH = 128
TARGET_HEIGH = 192

# Taken from https://github.com/nagadomi/lbpcascade_animeface
CLASSIFIER_FILE_NAME = 'lbpcascade_animeface.xml'

def get_avatar_rect(x, y, w, h, img_shape):
  img_height,  img_width, _ = img_shape
  x = max(x - w / 2, 0)
  w = min(w * 2, img_width)
  y = max(y - h / 2, 0)
  # Take more pixels below face to grep a bit of body.
  h = min(h * 3, img_height)
  return (x, y, w, h)


def process_file(src_file_path, dst_file_path, classifier):
  image = cv2.imread(src_file_path, cv2.IMREAD_COLOR)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  gray = cv2.equalizeHist(gray)

  faces, levels, weights = classifier.detectMultiScale3(gray,
                                      # detector options
                                      scaleFactor=1.1,
                                      minNeighbors=5,
                                      minSize=(24, 24),
                                      outputRejectLevels=True)
  if len(faces) == 0:
    return
  # Take one rect with maximum confidence level
  face_rect = faces[weights.argmax()]
  if not os.path.exists(os.path.dirname(dst_file_path)):
    os.makedirs(os.path.dirname(dst_file_path))
  x, y, w, h = face_rect
  x, y, w, h = get_avatar_rect(x, y, w, h, image.shape)
  face_img = image[y:y + h, x:x + w, :]
  if w < TARGET_WIDTH:
    interpolation_method = cv2.INTER_AREA
  else:
    interpolation_method = cv2.INTER_CUBIC
  face_img = cv2.resize(
      face_img,
      (TARGET_WIDTH, TARGET_HEIGH),
      interpolation=interpolation_method)
  cv2.imwrite(dst_file_path, face_img)


def get_src_file_paths(src_dir):
  for dirpath, _, filenames in os.walk(src_dir):
    for filename in filenames:
      if filename.endswith('.jpg'):
        yield os.path.join(dirpath, filename)



def process(downloader_output_dir, target_dir, classifier):
  downloader_output_dir = downloader_output_dir.rstrip(os.path.sep)
  target_dir = target_dir.rstrip(os.path.sep)
  file_paths = list(get_src_file_paths(downloader_output_dir))
  for src_file_path in tqdm.tqdm(file_paths, unit='image'):
    rel_path = src_file_path[len(downloader_output_dir) + 1:]
    dst_file_path = os.path.join(target_dir, rel_path)
    process_file(src_file_path, dst_file_path, classifier)


def main():
  if len(sys.argv) != 3:
    sys.stderr.write('Extracts avatar areas from downloaded '
        'images and scales them to fixed size\n');
    sys.stderr.write(
        'Usage: {} <downloader_output_dir> <dest dir>\n'.format(sys.argv[0]))
    sys.exit(1)
  classifier = cv2.CascadeClassifier(CLASSIFIER_FILE_NAME)
  process(sys.argv[1], sys.argv[2], classifier)


if __name__ == '__main__':
  main()

