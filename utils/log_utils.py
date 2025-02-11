import logging

def set_logging_config(path):
    logging.basicConfig(
         format='%(asctime)s %(levelname)-5s %(message)s',
         level=logging.INFO,
         datefmt='%Y-%m-%d %H:%M:%S',
         filename=path)