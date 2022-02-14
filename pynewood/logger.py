import datetime
import inspect
import sys


def who(n):
    """
    Get the class name and the method name of the caller to the log

    :param n: For current func name, specify 0 or no argument, for name of
    caller of current func, specify 1. For name of caller of caller of
    current func, specify 2. etc.
    :return: A string with the class name (if applicable, '<module>' otherwise
    and the calling function name.
    """
    max_stack_len = len(inspect.stack())
    depth = n + 1
    if (n + 1) > max_stack_len:
        depth = n
    calling_function = sys._getframe(depth).f_code.co_name
    if 'self' not in sys._getframe(depth).f_locals:
        class_name = 'NA'
        return calling_function
    else:
        class_name = sys._getframe(depth).f_locals["self"].__class__.__name__
        return '{}:{}'.format(class_name, calling_function)


class Logger:
    _DEBUG = 4
    _INFO = 3
    _WARN = 2
    _ERROR = 1
    _SILENT = 0

    # https://stackoverflow.com/questions/287871/print-in-terminal-with-colors
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    INFOGREY = '\033[30m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    nocolor = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    _level = 3

    def __init__(self, level=0):
        self._level = level
        if self._level > 3:
            print('Log level:', self._level, flush=True)

    def set_level(self, level):
        self._level = level
        self.info('Setting new log level to: {}'.format(level))

    @staticmethod
    def now():
        return '{date:%Y-%m-%d %H:%M:%S}'.format(date=datetime.datetime.now())

    def default_formatter(self, color, what, letter='', **kwargs):
        print('[{} {}] {}{}{} [@{}]'.format(
            letter, self.now(), color, what, self.nocolor, who(2), **kwargs))
        sys.stdout.flush()

    def debug(self, what, **kwargs):
        if self._level < self._DEBUG:
            return
        self.default_formatter(self.INFOGREY, what, 'D', **kwargs)

    def highlight(self, what, **kwargs):
        self.default_formatter(self.OKBLUE, what, 'i', **kwargs)

    def info(self, what, **kwargs):
        if self._level < self._INFO:
            return
        self.default_formatter(self.OKGREEN, what, 'I', **kwargs)

    def warn(self, what, **kwargs):
        if self._level < self._WARN:
            return
        self.default_formatter(self.WARNING, what, 'W', **kwargs)

    def error(self, what, **kwargs):
        if self._level < self._ERROR:
            return
        self.default_formatter(self.FAIL, what, 'E', **kwargs)