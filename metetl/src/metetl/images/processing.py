import os
import asyncio
import logging
import numpy as np
from io import BytesIO
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
import aiofiles

from metetl.images.models import Artwork


logger = logging.getLogger("metetl")

def gauss(image_data: np.ndarray) -> np.ndarray:
    logger.debug(f"Свёртка (Гаусс) началась (PID {os.getpid()})")
    core = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    np_result = np.zeros_like(image_data)
    for y in range(1, image_data.shape[0] - 1):
        for x in range(1, image_data.shape[1] - 1):
            window = image_data[y - 1:y + 2, x - 1:x + 2] * core[:, :, np.newaxis]
            np_result[y, x] = np.sum(window, axis=(0, 1)) / np.sum(core)
    res = np.clip(np_result, 0, 255).astype(np.uint8)
    logger.debug("Свёртка (Гаусс) завершилась")
    return res

def sobel(image_data: np.ndarray) -> np.ndarray:
    logger.debug(f"Свёртка (Собель) началась (PID {os.getpid()})")
    g_v = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    g_h = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gray = (0.2126 * image_data[:,:,0] + 0.7152 * image_data[:,:,1] + 0.0722 * image_data[:,:,2]).astype(np.uint8)
    np_result = np.zeros_like(gray)
    for y in range(1, gray.shape[0] - 1):
        for x in range(1, gray.shape[1] - 1):
            window = gray[y - 1:y + 2, x - 1:x + 2]
            np_result[y, x] = np.sqrt(np.sum((window * g_h) ** 2 + (window * g_v) ** 2))
    res = np.clip(np_result, 0, 255).astype(np.uint8)
    logger.debug("Свёртка (Собель) завершилась")
    return res

class ImageProcessor:
    def __init__(self, save_directory: str, executor: ProcessPoolExecutor):
        self.save_directory = save_directory
        self.executor = executor
        os.makedirs(save_directory, exist_ok=True)

    async def download_painting(self, session, info: dict, index: int) -> Artwork:
        obj_id = info.get('objectID', 'unknown')
        logger.info(f"Скачивание изображения №{index} (ID: {obj_id})")
        try:
            async with session.get(info['primaryImage'], timeout=30) as response:
                if response.status != 200:
                    logger.error(f"Не удалось скачать ID {obj_id}: Статус {response.status}")
                    return None
                image_bytes = await response.read()
            
            np_image = np.array(Image.open(BytesIO(image_bytes)).convert('RGB'))
            await self.save_image(np_image, f"{index}_{obj_id}_original")
            return Artwork(info, np_image, index)
        except Exception as e:
            logger.error(f"Ошибка при скачивании ID {obj_id}: {e}")
            return None

    async def process_artwork(self, artwork: Artwork):
        if artwork is None:
            return
        loop = asyncio.get_event_loop()
        p_id = artwork.metadata.get('objectID', 'unknown')
        idx = artwork.order_index

        await self.save_image(artwork.halftone(), f"{idx}_{p_id}_halftone")
        
        g_res = await loop.run_in_executor(self.executor, gauss, artwork.image)
        await self.save_image(g_res, f"{idx}_{p_id}_gauss")

        s_res = await loop.run_in_executor(self.executor, sobel, artwork.image)
        await self.save_image(s_res, f"{idx}_{p_id}_sobel")

    async def save_image(self, ndarray: np.ndarray, name: str) -> None:
        path = os.path.join(self.save_directory, f"{name}.png")
        img = Image.fromarray(ndarray)
        buf = BytesIO()
        img.save(buf, format='PNG')
        async with aiofiles.open(path, mode='wb') as f:
            await f.write(buf.getvalue())