import sys
import logging
from metetl.logging_config import setup_logging
from metetl.cli import parse_arguments, run_command

def main():
    setup_logging()
    
    logger = logging.getLogger("metetl")
    logger.info("Программа MetETL успешно запущена.")
    
    try:
        args = parse_arguments()
        run_command(args)
        logger.info("Программа успешно завершила свою работу.")
        
    except KeyboardInterrupt:
        logger.warning("Выполнение программы было прервано пользователем (Ctrl+C).")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Критическая ошибка при выполнении программы: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()