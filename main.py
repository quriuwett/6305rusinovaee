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

def gauss(np_array):
    core = np.array([
        [1,2,1],
        [2,4,2],
        [1,2,1]
    ])

    np_result = np.zeros_like(np_array)

    for y in range(1, np_array.shape[0] - 1):
        for x in range(1,np_array.shape[1] - 1):
                window = np_array[y-1:y+2, x-1:x+2] * core[:, :, np.newaxis]
                np_result[y,x] = np.sum(window, axis=(0,1)) / np.sum(core)
    return np.clip(np_result, 0, 255).astype(np.uint8)

def sobel(np_array):
    g_v = np.array([
        [-1,-2,-1],
        [0,0,0],
        [1,2,1]
    ])
    g_h = np.array([
        [-1,0,1],
        [-2,0,2],
        [-1,0,1]
    ])

    np_array = halftone(np_array)
    np_result = np.zeros_like(np_array)

    for y in range(1, np_array.shape[0] - 1):
        for x in range(1,np_array.shape[1] - 1):
                window = np_array[y-1:y+2, x-1:x+2]
                np_result[y,x] = np.sqrt(np.sum((window * g_h)**2 + (window * g_v)**2))
    return np.clip(np_result, 0, 255).astype(np.uint8)

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

#gauss_np_image = gauss(np_image)
#save_np(directory,'gauss_image.jpg', gauss_np_image)

sobel_np_image = sobel(np_image)
save_np(directory,'sobel_image.jpg', sobel_np_image)