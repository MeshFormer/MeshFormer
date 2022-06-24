import loguru as log
import os
import sys

import loguru as log
import os
import sys
import time



class Logger:
    def __init__(self, path):
        lpath = os.path.join(path, 'file_'+str(time.time()))
        self.log = log.logger
        self.log.remove()  # Remove the default setting
        self.log.add(lpath)
        self.log.add(sys.stdout, colorize=True, format="<green>{time}</green> <level>{message}</level>")
        self.setup_logger()

    def debug(self,meg):
        self.log.debug(meg)
        # self.log.add(sys.stdout,  format="{time} {level} {message}", filter="my_module", level="INFO")
    def info(self,info):
        self.log.info(info)

    def error(self,error):
        self.log.error(error)
    def log(self,log):
        self.log.log(log)

    def setup_logger(self) -> None:
        """Set up stderr logging format.

        The logging format and colors can be overridden by setting up the
        environment variables such as ``LOGURU_FORMAT``.
        See `Loguru documentation`_ for details.

        .. _Loguru documentation: https://loguru.readthedocs.io/en/stable/api/logger.html#env
        """
        # Set up the preferred logging colors and format unless overridden by its environment variable
        self.log.level("INFO", color= "<white>")
        self.log.level("DEBUG", color="<d><white>")
        log_format =  (
            # "<green>{time:YYYY-MM-DD HH:mm:ss}</green> "
            "<b><level>{level: <8}</level></b> "
            "| <level>{message}</level>"
        )
        self.log.enable("charger")



