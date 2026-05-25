import argparse
import asyncio
import json
import logging
from concurrent.futures import ProcessPoolExecutor
import aiohttp

from metetl.analysis.data_to_download import prepare_metadata
from metetl.analysis.aggregations import run_analysis
from metetl.images.processing import ImageProcessor

logger = logging.getLogger("metetl")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Met Museum ETL Tool")
    subparsers = parser.add_subparsers(dest="command", help="Доступные команды")

    prep = subparsers.add_parser("prepare", help="Подготовка JSON с метаданными")
    prep.add_argument("--csv", required=True, help="Путь к исходному CSV файлу")
    prep.add_argument("--output", required=True, help="Путь для сохранения JSON")
    prep.add_argument("--num", type=int, default=5, help="Количество изображений")

    proc = subparsers.add_parser("process", help="Скачивание и обработка")
    proc.add_argument("--input", required=True, help="Путь к JSON с метаданными")
    proc.add_argument("--output", required=True, help="Директория для сохранения")
    proc.add_argument("--num", type=int, default=5, help="Количество для обработки")

    anlz = subparsers.add_parser("analyze", help="Анализ данных")
    anlz.add_argument("--csv", required=True, help="Путь к исходному CSV файлу")
    anlz.add_argument("--output-dir", required=True, help="Директория для графиков")

    return parser.parse_args()

def run_command(args):
    if args.command == "prepare":
        asyncio.run(prepare_metadata(args.csv, args.output, args.num))
    
    elif args.command == "process":
        asyncio.run(run_process(args.input, args.output, args.num))
        
    elif args.command == "analyze":
        run_analysis(args.csv, args.output_dir)
    else:
        logger.warning("Команда не указана или не распознана.")
        argparse.ArgumentParser(description="Met Museum ETL Tool").print_help()

async def run_process(input_json, output_dir, num):
    logger.info("Запуск пайплайна обработки изображений...")
    
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)[:num]

    if not data:
        logger.warning(f"Файл {input_json} пуст или не содержит данных.")
        return

    with ProcessPoolExecutor() as executor:
        processor = ImageProcessor(output_dir, executor)
        async with aiohttp.ClientSession() as session:
            logger.info(f"Начало асинхронного скачивания {len(data)} изображений...")
            tasks = [processor.download_painting(session, item, i+1) for i, item in enumerate(data)]
            artworks = await asyncio.gather(*tasks, return_exceptions=True)
            
            valid_artworks = [art for art in artworks if not isinstance(art, Exception) and art is not None]

            logger.info("Начало многопроцессорной обработки (Гаусс/Собель)...")
            proc_tasks = [processor.process_artwork(art) for art in valid_artworks]
            await asyncio.gather(*proc_tasks)
            
    logger.info("Обработка всех изображений завершена.")