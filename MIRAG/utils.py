import logging
import time
from tqdm import tqdm
import io
import sys



class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    Fetched from https://stackoverflow.com/questions/19425736/how-to-redirect-stdout-and-stderr-to-logger-in-python
    Access date: 10 January 2023 - 19:19
    """
    def __init__(self, logger, level):
       self.logger = logger
       self.level = level
       self.linebuf = ''

    def write(self, buf):
       for line in buf.rstrip().splitlines():
          self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass

class TqdmToLogger(io.StringIO):
    """
        Output stream for TQDM which will output to logger module instead of
        the StdOut.
        Fetched from: https://stackoverflow.com/questions/14897756/python-progress-bar-through-logging-module.
        Access date: 10 January 2023 - 18:30
    """
    logger = None
    level = None
    buf = ''
    def __init__(self,logger,level=None):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO
    def write(self,buf):
        self.buf = buf.strip('\r\n\t ')
    def flush(self):
        self.logger.log(self.level, self.buf)