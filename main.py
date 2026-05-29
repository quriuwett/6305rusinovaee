import cv2
import os
from csv import DictReader
from io import BytesIO
from random import choice
from typing import Dict, List
import numpy as np
from numpy.typing import NDArray
import time
from requests import get
from PIL import Image

def measure_time(name: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            print(f'Функция "{name}" выполнилась за {end - start} сек')
            return result
        return wrapper
    return decorator

class Artwork:
    __slots__ = ['__image', '__metadata']

    def __init__(self, metadata: Dict[str, str], image: NDArray[np.uint8]):
        self.__metadata = metadata
        self.__image = image

    @property
    def metadata(self) -> Dict[str, str]:
        return self.__metadata

    @property
    def image(self) -> NDArray[np.uint8]:
        return self.__image

    @measure_time('Полутонирование')
    def halftone(self) -> NDArray[np.uint8]:
        res = (np.array(0.2126 * self.__image[:, :, 0]
                       + 0.7152 * self.__image[:, :, 1]
                       + 0.0722 * self.__image[:, :, 2])
        .astype(np.uint8))
        return res

    @measure_time('Размытие по Гауссу')
    def gauss(self) -> NDArray[np.uint8]:
        core = np.array([
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ])
        np_result = np.zeros_like(self.__image)

        for y in range(1, self.__image.shape[0] - 1):
            for x in range(1, self.__image.shape[1] - 1):
                window = self.__image[y - 1:y + 2, x - 1:x + 2] * core[:, :, np.newaxis]
                np_result[y, x] = np.sum(window, axis=(0, 1)) / np.sum(core)
        res = np.clip(np_result, 0, 255).astype(np.uint8)
        return res

    @measure_time('Выделение границ Собеля')
    def sobel(self) -> NDArray[np.uint8]:
        g_v = np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ])
        g_h = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ])

        np_array = self.halftone()
        np_result = np.zeros_like(np_array)

        for y in range(1, np_array.shape[0] - 1):
            for x in range(1, np_array.shape[1] - 1):
                window = np_array[y - 1:y + 2, x - 1:x + 2]
                np_result[y, x] = np.sqrt(np.sum((window * g_h) ** 2 + (window * g_v) ** 2))
        res = np.clip(np_result, 0, 255).astype(np.uint8)
        return res

    def _resize_to_max(self, other: 'Artwork'):
        h1, w1 = self.__image.shape[:2]
        h2, w2 = other.__image.shape[:2]
        
        target_w = max(w1, w2)
        target_h = max(h1, h2)
        target_size = (target_w, target_h)

        img1_resized = cv2.resize(self.__image, target_size, interpolation=cv2.INTER_LANCZOS4)
        img2_resized = cv2.resize(other.__image, target_size, interpolation=cv2.INTER_LANCZOS4)

        return img1_resized, img2_resized

    def __add__(self, other: 'Artwork') -> 'Artwork':
        img1, img2 = self._resize_to_max(other)
        combined = (img1.astype(np.float32) + img2.astype(np.float32)) / 2
        new_meta = {**self.__metadata, 'title': f"Смесь: {self.__metadata.get('title', 'Art1')} + {other.metadata.get('title', 'Art2')}"}
        return Artwork(new_meta, combined.astype(np.uint8))

    def __sub__(self, other: 'Artwork') -> 'Artwork':
        img1, img2 = self._resize_to_max(other)
        diff = np.abs(img1.astype(np.int16) - img2.astype(np.int16))
        new_meta = {**self.__metadata, 'title': f"Разность: {self.__metadata.get('title', 'Art1')} - {other.metadata.get('title', 'Art2')}"}
        return Artwork(new_meta, diff.astype(np.uint8))

    def __str__(self) -> str:
        return self.__metadata['title']


class ImageProcessor:
    __api_url = 'https://collectionapi.metmuseum.org/public/collection/v1/objects/'

    def __init__(self, csv_path: str, save_directory: str):
        self.__csv_path = csv_path
        self.__save_directory = save_directory

    def download_painting(self, name: str) -> Artwork:
        print('Скачивание данных об изображении...')
        paintings = []
        with open(self.__csv_path, encoding='utf-8') as f:
            reader = DictReader(f)
            for m in reader:
                if m['Classification'] == 'Paintings':
                    paintings.append(m)

        info = self.__download_info(paintings)
        while info['primaryImage'] == '':
            info = self.__download_info(paintings)


        image = get(info['primaryImage']).content
        np_image = np.array(Image.open(BytesIO(image)).convert('RGB'))
        print('Скачивание завершено')
        self.save_image(np_image, name)
        return Artwork(info, np_image)

    def __download_info(self, paintings: List[int]) -> Dict[str, str]:
        painting = choice(paintings)
        object_id = painting['Object ID']
        info = get(self.__api_url + object_id)
        return info.json()

    def sobel(self, artwork: Artwork, save_name: str = 'sobel') -> None:
        print('Применение фильтра Собеля...')
        res = artwork.sobel()
        print('Сохранение...')
        self.save_image(res, save_name)
        print(f'Изображение "{save_name}" сохранено')

    def gauss(self, artwork: Artwork, save_name: str = 'gauss') -> None:
        print('Применение размытия Гаусса...')
        res = artwork.gauss()
        print('Сохранение...')
        self.save_image(res, save_name)
        print(f'Изображение "{save_name}" сохранено')

    def halftone(self, artwork: Artwork, save_name: str = 'halftone') -> None:
        print('Полутонирование изображения...')
        res = artwork.halftone()
        print('Сохранение...')
        self.save_image(res, save_name)
        print(f'Изображение "{save_name}" сохранено')

    def save_image(self, ndarray: NDArray[np.uint8], name: str) -> None:
        path = os.path.join(self.__save_directory, f"{name}.jpg")
        img = Image.fromarray(ndarray)
        img.save(path)

    
def main():
    CSV_FILE = 'C:\\Users\\HP\\Desktop\\MetObjects.csv'
    SAVE_DIR = 'paintings'

    processor = ImageProcessor(csv_path=CSV_FILE, save_directory=SAVE_DIR)
    art = processor.download_painting('image')

    processor.halftone(art, "grayscale")
    processor.gauss(art, "blurred")
    processor.sobel(art, "edges")

    art1 = processor.download_painting('art1')
    art2 = processor.download_painting('art2')
    added_art = art1 + art2
    processor.save_image(added_art.image, 'added')

    subbed_art = art1 - art2
    processor.save_image(subbed_art.image, 'subbed')

    

if __name__ == "__main__":
    main()