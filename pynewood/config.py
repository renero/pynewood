"""
This class reads params from a YAML file and creates an object that
contains attributes named as the params in the file, accessible through
getters:

  object.parameter

in addition to classical dictionary access method

  object[parameter]

The structure of attributes is built recursively if they contain a dictionary.

  object.attr1.attr2.attr3

"""
from collections import defaultdict
from os import getcwd
from pathlib import Path

from yaml import YAMLError, safe_load

from .logger import Logger
from .utils import dict2table

debug = False


class Configuration(defaultdict):

    def __init__(self):
        # https://stackoverflow.com/a/45411093/892904
        super(Configuration, self).__init__()

    def __getattr__(self, key):
        """
        Check out https://stackoverflow.com/a/42272450
        """
        if key in self:
            return self.get(key)
        raise AttributeError('Key <{}> not present in dictionary'.format(key))

    def __setattr__(self, key, value):
        self[key] = value

    def __str__(self):
        return dict2table(self)

    @staticmethod
    def logdebug(*args, **kwargs):
        if debug is True:
            print(*args, **kwargs)

    def add_dict(self, this_object, param_dictionary, add_underscore=True):
        for param_name in param_dictionary.keys():
            self.logdebug('ATTR: <{}> type is {}'.format(
                param_name, type(param_dictionary[param_name])))

            if add_underscore is True:
                attribute_name = '{}'.format(param_name)
            else:
                attribute_name = param_name

            if type(param_dictionary[param_name]) is not dict:
                self.logdebug(' - Setting attr name {} to {}'.format(
                    attribute_name, param_dictionary[param_name]))
                setattr(this_object, attribute_name,
                        param_dictionary[param_name])
            else:
                self.logdebug(' x Dictionary Found!')

                self.logdebug('   > Creating new dict() with name <{}>'.format(
                    attribute_name))
                setattr(this_object, attribute_name, Configuration())

                self.logdebug('     > New Attribute <{}> type is: {}'.format(
                    attribute_name, type(getattr(this_object, attribute_name))
                ))
                new_object = getattr(this_object, attribute_name)

                self.logdebug('   > Calling recursively with dict')
                self.logdebug('     {}'.format(param_dictionary[param_name]))
                this_object.add_dict(new_object, param_dictionary[param_name])


def config(params_filename=".pynewood.yaml", log_level=None) -> \
        (Configuration, Logger):
    """
    Read the parameters from a filename, and returns a dictionary (default)
    and a basic logger.

    Args:
    params_filename: the name of the YAML file you want to use as source
        of parameters. If none specified, a file called ".pynewood.yaml" is
        searched for in local directory first, and then in home folder.
    log_level: an integer between 0 (silent) and 4 (debug=totally verbose)
        specifying the logging level. This parameter is overriden by the value
        in the YAML configuration file. Default value if not specified is
        3 = WARNING.

    Returns:
    A customdict object containing the parameters read from file, and a Logger.
    """
    configuration = Configuration()
    local_yaml = False
    home_yaml = False
    bad_yaml = False
    no_yaml = False

    try:
        cwd = Path(getcwd())
        params_path: str = str(cwd.joinpath(params_filename))
        with open(params_path, 'r') as stream:
            try:
                params_read = safe_load(stream)
                configuration.add_dict(configuration, params_read)
                local_yaml = True
            except YAMLError:
                bad_yaml = True
                pass
    except FileNotFoundError:
        home = Path.home()
        params_path: str = str(home.joinpath(params_filename))
        try:
            with open(params_path, 'r') as stream:
                try:
                    params_read = safe_load(stream)
                    configuration.add_dict(configuration, params_read)
                    home_yaml = True
                except YAMLError:
                    bad_yaml = True
                    pass
        except FileNotFoundError:
            no_yaml = True
            pass

    #
    # Set log_level and start the logger
    #
    if 'log_level' not in configuration:
        if log_level is not None:
            configuration.log_level = log_level  # as specified
        else:
            configuration.log_level = 3  # default value = WARNING

    log = Logger(configuration.log_level)
    if no_yaml:
        log.warn(f"WARNING: No {params_filename} parameters file.")
    if bad_yaml:
        log.warn('Bad formatted YAML!')
    if home_yaml:
        log.info("Home folder YAML file loaded")
    if local_yaml:
        log.info("Local folder YAML file loaded")

    return configuration, log
