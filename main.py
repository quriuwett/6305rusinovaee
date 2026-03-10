from csv import DictReader
from random import choice
from requests import get
from os import path, makedirs
import shutil
import json
import numpy as np
from PIL import Image

def download_info(paintings, api_url):
    painting = choice(paintings)
    object_id = painting['Object ID']
    info = get(api_url + object_id)
    return info.json()

def halftone(np_array):
    return np.array(0.2126 * np_array[:,:,0] + 0.7152 * np_array[:,:,1] + 0.0722 * np_array[:,:,2]).astype(np.uint8)

def save_np(directory, file_name, np_array):
    array_path = path.join(directory, file_name)
    image = Image.fromarray(np_array)
    image.save(array_path)

paintings = []
api_url = 'https://collectionapi.metmuseum.org/public/collection/v1/objects/'
directory = 'paintings'

with open('MetObjects.csv', encoding='utf-8') as f:
    reader = DictReader(f)
    for map in reader:
        if map['Classification'] == 'Paintings':
            paintings.append(map)

info = download_info(paintings, api_url)
while info['primaryImage'] == '':
    info = download_info(paintings, api_url)

image_url = info['primaryImage']
image = get(image_url).content

if path.exists(directory):
    shutil.rmtree(directory)
    makedirs(directory)
else:
    makedirs(directory)

image_path = path.join(directory, 'img.jpg')
info_path = path.join(directory, 'info.json')

with open(image_path, 'wb') as f:
    f.write(image)
with open(info_path, 'w') as f:
    json.dump(info, f)

np_image = np.array(Image.open(image_path).convert('RGB'))
halftone_np_image = halftone(np_image)
save_np(directory, 'halftone_image.jpg', halftone_np_image)