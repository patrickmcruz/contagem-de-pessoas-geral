import logging
import os
import sys
from datetime import datetime

class CustomFormatter(logging.Formatter):
    """Formatador customizado para seguir o padrão solicitado: [YYYY-MM-DD HH:MM:SS.SSS] [LEVEL]|"""
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created)
        if datefmt:
            s = dt.strftime(datefmt)
        else:
            s = dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] # Trunca para milissegundos
        return s

    def format(self, record):
        log_fmt = "[%(asctime)s] [%(levelname)s]| %(message)s"
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S.%f")
        # Ajusta manualmente os milissegundos no final
        formatted_time = self.formatTime(record)
        record.asctime = formatted_time
        return super(CustomFormatter, self).format(record)

def setup_logger(log_level="DEBUG", log_dir="logs", filename="execution.log"):
    """Configura o logger global com arquivos e console."""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_path = os.path.join(log_dir, filename)
    
    # Criar logger raiz
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Limpar handlers existentes (evitar duplicatas no reload)
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formatter customizado
    formatter = CustomFormatter("[%(asctime)s] [%(levelname)s]| %(message)s")

    # Handler de Arquivo
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    # Handler de Console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
