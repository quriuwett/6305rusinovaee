import time
import logging

logger = logging.getLogger("metetl")

def measure_time(name: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            logger.debug(f'Функция "{name}" выполнилась за {end - start:.4f} сек')
            return result
        return wrapper
    return decorator