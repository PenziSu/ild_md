import logging
import os

class LogUtility():
    def __init__(self, log_filename):
        self.logger = logging.getLogger(log_filename)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_filename)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def info(self, msg):
        self.logger.info(msg)

    def error(self, msg):
        self.logger.error(msg)