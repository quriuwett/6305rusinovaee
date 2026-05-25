import csv
import json
import random
import logging
import aiohttp

logger = logging.getLogger("metetl")

async def prepare_metadata(csv_path: str, output_json: str, count: int = 10):
    logger.info(f"Подготовка метаданных из {csv_path}")
    paintings = []
    with open(csv_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('Classification') == 'Paintings' and row.get('Object ID'):
                paintings.append(row)
    
    if not paintings:
        logger.warning("В исходном файле не найдено объектов с классификацией 'Paintings'")
        return

    selected = random.sample(paintings, min(len(paintings), count * 2))
    results = []
    
    api_url = 'https://collectionapi.metmuseum.org/public/collection/v1/objects/'
    async with aiohttp.ClientSession() as session:
        for item in selected:
            if len(results) >= count: 
                break
            try:
                async with session.get(api_url + str(item['Object ID']), timeout=15) as resp:
                    if resp.status == 200:
                        info = await resp.json()
                        if info.get('primaryImage'):
                            results.append(info)
                            logger.debug(f"Найдено изображение для ID {item['Object ID']}")
            except Exception as e:
                logger.error(f"Ошибка запроса к API для ID {item['Object ID']}: {e}")

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    logger.info(f"Файл {output_json} успешно создан. Найдено объектов: {len(results)}")