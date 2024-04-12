import logging
import sys
import os


def get_logger(name, level, fmt='%(asctime)s - %(name)s - %(levelname)s: %(message)s'):
    """
    Get logger from logging with given name, level and format without
    setting logging basicConfig. For setting basicConfig in paddle
    will disable basicConfig setting after import paddle.

    Args:
        name (str): The logger name.
        level (logging.LEVEL): The base level of the logger
        fmt (str): Format of logger output, such as:
            '%(levelname)s: %(message)s'
            '%(asctime)s - %(levelname)s: %(message)s'
            '%(asctime)s - %(name)s - %(levelname)s: %(message)s'


    Returns:
        logging.Logger: logging logger with given setttings

    Examples:

    .. code-block:: python

       logger = get_logger(__name__, logging.INFO,
                            fmt='%(asctime)s-%(levelname)s: %(message)s')
    """
    if isinstance(level, str):
        level_dict = {'info': logging.INFO,
                      'debug': logging.DEBUG,
                      'warning': logging.WARNING,
                      'error': logging.ERROR}
        level = level_dict[level]
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = logging.StreamHandler()
    if fmt:
        formatter = logging.Formatter(fmt=fmt)
        handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = 0
    return logger


class LoggerWrite(object):
    """
        write terminal info or error to file
    """
    def __init__(self, filename, stream=sys.stdout, mode='a'):
        self.terminal = stream
        self.log = open(filename, mode)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


class Logger(object):
    def __init__(self, name, saving_path = '', level='info', fmt='default'):
        self.levels = {
            'info': logging.INFO,
            'debug': logging.DEBUG,
            'warning': logging.WARN,
        }
        self.fmts = {
            'default': '%(message)s',
            'level': '%(levelname)s: %(message)s',
            'time': '%(asctime)s - %(levelname)s: %(message)s',
            'full': '%(asctime)s - %(name)s - %(levelname)s: %(message)s'
        }
        self.colors = {
            "black": 30,
            "red": 31,
            "green": 32,
            "yellow": 33,
            "blue": 34,
            "purple": 35,
            "cyan": 36,
            "white": 37
        }
        if saving_path!='':
            os.makedirs(saving_path, exist_ok=True)
            sys.stdout = LoggerWrite(f'{saving_path}/log_stdout.log', sys.stdout, 'a')
            sys.stderr = LoggerWrite(f'{saving_path}/log_stderr.log', sys.stderr, 'a')
        self.logger = self.get_logger(name, saving_path=saving_path, level=self.levels[level], fmt=fmt)

    def get_logger(self, name, saving_path = '', level=logging.INFO, fmt='levelname'):
        logger = logging.getLogger(name)
        logger.setLevel(level)
        handler = logging.StreamHandler()
        if fmt:
            formatter = logging.Formatter(fmt=self.fmts[fmt])
            handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = 0

        if saving_path!='':
            info_file_handler = logging.FileHandler(f'{saving_path}/log_logger.log')
            info_file_handler.setLevel(level)
            logger.addHandler(info_file_handler)
        return logger

    def apply_color(self, msg, color):
        return "\033[%dm%s\033[0m" % (self.colors[color], msg)

    def info(self, msg):
        self.logger.info(msg)

    def debug(self, msg, is_color=False):
        msg = self.apply_color(msg, 'green') if is_color else msg
        self.logger.debug(msg)

    def warning(self, msg, is_color=False):
        msg = self.apply_color(msg, 'yellow') if is_color else msg
        self.logger.warning(msg)

    def error(self, msg, is_color=False):
        msg = self.apply_color(msg, 'red') if is_color else msg
        self.logger.error(msg)

    def progress_bar(self, current, total, msg=None):
        TOTAL_BAR_LENGTH = 30.
        cur_len = int(TOTAL_BAR_LENGTH * current / total)
        rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1
        sys.stdout.write('[')
        for i in range(cur_len):
            sys.stdout.write('=')
        sys.stdout.write('>')
        for i in range(rest_len):
            sys.stdout.write('.')
        sys.stdout.write(']')
        sys.stdout.write(msg)
        if current < total - 1:
            sys.stdout.write('\r')
        else:
            sys.stdout.write('\n')
        sys.stdout.flush()

if __name__ == '__main__':
    # logger = Logger('Test', saving_path = '', level='info', fmt='level')
    # logger = Logger('Test', saving_path = '', level='warning', fmt='level')
    logger = Logger('Test', saving_path = './', level='debug', fmt='full')
    # logger = Logger('Test', saving_path = '', level='debug', fmt='full')

    logger.info('info')
    logger.debug('debug', True)
    logger.warning('warning+++++++++', True)  # NOTE when with color, the log saved will be different
    logger.debug('debug', False)
    logger.warning('warning+++++++++', False)
    logger.progress_bar(10, 100, 'bar')