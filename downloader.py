#!/usr/bin/env python

import bs4
import csv
import os
import re
import sys
import tqdm
import urllib2
import urlparse

def download_url(src_url):
  request = urllib2.Request(src_url)
  request.add_header('Cookie', 'getchu_adalt_flag=getchu.com')
  f_request = urllib2.urlopen(request)
  return f_request.read()

def get_image_urls(page_data, game_id):
  parsed_page = bs4.BeautifulSoup(page_data, "lxml")
  imgs = parsed_page.findAll('img', src=True)
  re_character_url = re.compile(
      'http://www.getchu.com/brandnew/{game_id}/[^/]+chara[^/]+'.format(
          game_id=game_id))
  result = []
  for img in imgs:
    url = urlparse.urljoin('http://www.getchu.com/', img['src'])
    if re_character_url.match(url):
      result.append(url)
  return result

def download_for_game_id(game_id, dest_dir):
  game_dir = os.path.join(dest_dir, str(game_id))
  src_url = 'http://www.getchu.com/soft.phtml?id={}'.format(game_id)
  page_data = download_url(src_url)
  img_urls = get_image_urls(page_data, game_id)
  if len(img_urls) > 0:
    if not os.path.exists(game_dir):
      os.makedirs(game_dir)
  for index, img_url in enumerate(img_urls):
    img_data = download_url(img_url)
    img_file_path = os.path.join(game_dir, '{}.jpg'.format(index))
    with open(img_file_path, 'wb') as f_out:
      f_out.write(img_data)

def parse_csv(csv_file_path):
  result = []
  with open(csv_file_path, 'rb') as f:
    reader = csv.DictReader(f, delimiter=',', quotechar='"')
    for row in reader:
      result.append(row['links'])
  return result

def download_data(csv_file_path, dest_dir):
  game_ids = parse_csv(csv_file_path)
  for game_id in tqdm.tqdm(game_ids, unit='page'):
    download_for_game_id(game_id, dest_dir)

def main():
  # CSV file should be obtained through executing query
  #
  # SELECT g.id, g.gamename, g.sellday, g.comike as links
  # FROM gamelist g
  # WHERE g.comike is NOT NULL
  # ORDER BY g.sellday
  #
  # on http://erogamescape.dyndns.org/~ap2/ero/toukei_kaiseki/sql_for_erogamer_form.php

  if len(sys.argv) != 3:
    sys.stderr.write('Downloads character images from www.getchu.com\n')
    sys.stderr.write('Usage: {} <db_csv_file> <dest dir>\n'.format(sys.argv[0]))
    sys.exit(1)
  download_data(sys.argv[1], sys.argv[2])

if __name__ == '__main__':
  main()
