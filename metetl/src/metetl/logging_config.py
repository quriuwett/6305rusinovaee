import os
import json
import logging
import logging.config

def setup_logging():
    os.makedirs("./logs", exist_ok=True)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "logging_config.json")
    
    if not os.path.exists(config_path):
        logging.basicConfig(level=logging.INFO)
        logging.warning(f"Файл конфигурации логов не найден по пути: {config_path}. Используются настройки по умолчанию.")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
        
    logging.config.dictConfig(config_dict)