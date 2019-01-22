import os
import logging
from logging import config

from utils import getPath, readYaml
# from logging.handlers import RotatingFileHandler



class Logger():
    def __init__(self, logFile='f1'):
        self.logFile = logFile
        self.logConfigFile = getPath()['LOG_CONFIG_FILE']
        self.setLogger()    
    def setLogger(self):
        D = readYaml(self.logConfigFile)
        logging.config.dictConfig(D)
        
        
        self.logger = logging.getLogger(__name__)
        # handler = logging.FileHandler(self.logFile)
        # self.logger.addHandler(handler)

    def getLogger(self):
        return self.logger
