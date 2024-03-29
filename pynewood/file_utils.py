import errno
import glob
import json
import os
import pickle
from os.path import dirname, realpath, join
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler


def file_exists(given_filepath: str, my_dir: str) -> bool:
    """
    Check if the file exists as specified in argument, or try to find
    it using the local path of the script

    :param given_filepath:
    :return: Whether the file has been found.
    """
    if os.path.exists(given_filepath) is True:
        return True
    else:
        new_filepath = os.path.join(my_dir, given_filepath)
        if os.path.exists(new_filepath) is True:
            return True
        else:
            return False


def locate_file(given_filepath: str, my_dir: str) -> str:
    """
    Check if the file exists as specified in argument, or try to find
    it using the local path of the script
    :param given_filepath:
    :return: The path where the file is or None if it couldn't be found
    """
    if os.path.exists(given_filepath) is True:
        filepath = given_filepath
    else:
        new_filepath = os.path.join(my_dir, given_filepath)
        if os.path.exists(new_filepath) is True:
            filepath = new_filepath
        else:
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), new_filepath)
    print(f'filepath = {filepath}')
    return filepath


def valid_output_name(filename: str, path: str, extension=None) -> str:
    """
    Builds a valid name. In case there's another file which already exists
    adds a number (1, 2, ...) until finds a valid filename which does not
    exist.

    Returns
    -------
    The filename if the name is valid and file does not exists,
            None otherwise.

    Params
    ------
    filename: str
        The base filename to be set.
    path: str
        The path where trying to set the filename
    extension:str
        The extension of the file, without the dot '.' If no extension is
        specified, any extension is searched to avoid returning a filepath
        of an existing file, no matter what extension it has.
    """
    #path = file_exists(path, dirname(realpath(__file__)))
    if extension:
        base_filepath = join(path, filename) + '.{}'.format(extension)
    else:
        base_filepath = join(path, filename)
    output_filepath = base_filepath
    idx = 1
    while len(glob.glob(f"{output_filepath}*")) > 0:
        if extension:
            output_filepath = join(
                path, filename) + '_{:d}.{}'.format(
                idx, extension)
        else:
            output_filepath = join(path, filename + '_{}'.format(idx))
        idx += 1

    #print(f'output_filepath: {output_filepath}')
    return output_filepath


def change_extension_to(original_file: str, new_extension: str) -> str:
    """
    Given a filename, returns a new filename with the same basename but with a
    different extension, specified in argument.

    Args:
          original_file: (str) the filename change extension
          new_extension: (str) the new extension to be used with the filename
                               Do not use dots in the new extension.

    Returns:
          str: the filename with the original extension replaced with the new
    """
    if "." in new_extension and new_extension.indes(".") == 0:
        new_extension = new_extension[1:]
    try:
        pos = original_file.index('.')
    except ValueError:
        return original_file + "." + new_extension
    if pos == original_file.rindex("."):
        return original_file[:pos] + "." + new_extension
    else:
        pos = original_file.rindex(".")
        return original_file[:pos] + "." + new_extension


def save_dataframe(name: str,
                   df: DataFrame,
                   output_path: str,
                   cols_to_scale: list = None,
                   scaler_name: str = None,
                   index: bool = True) -> Tuple[str, str]:
    """
    Save the data frame passed, with a valid output name in the output path
    scaling the columns specified, if applicable.

    :param name: the name to be used to save the df to file
    :param df: the dataframe
    :param output_path: the path where the df is to be saved
    :param cols_to_scale: array with the names of the columns to scale
    :param scaler_name: baseName of the file where saving the scaler used.
    :param index: save index in the csv

    :return: the full path of the file saved
    """
    data = df.copy()
    file_name = valid_output_name(name, output_path, 'csv')
    if cols_to_scale is not None:
        scaler = MinMaxScaler(feature_range=(-1., 1.))
        data[cols_to_scale] = scaler.fit_transform(data[cols_to_scale])
        # Save the scaler used
        scaler_name = valid_output_name(scaler_name, output_path, 'pickle')
        joblib.dump(scaler, scaler_name)
    data.round(4).to_csv(file_name, index=index)

    return file_name, scaler_name


def scale(x, minimum, peak_to_peak):
    return (x - minimum) / peak_to_peak


def unscale(y, minimum, peak_to_peak):
    return (peak_to_peak * y) + minimum


def scale_columns(df, mx):
    """
    Scale columns from a dataframe to 0..1 range.

    :param df:              the dataframe
    :param mn:              the minimum value of the series
    :param mx:              the maximum value of the series

    :return: The data frame scaled
    """
    df = df.apply(lambda x: scale(x, minimum=0., peak_to_peak=mx - 0.))
    return df


def unscale_columns(df, mx):
    """
    UN_Scale columns from a dataframe to 0..1 range.

    :param df:              the dataframe
    :param mn:              the minimum value of the series
    :param mx:              the maximum value of the series

    :return: The data frame scaled
    """
    df = df.apply(lambda y: unscale(y, minimum=0.0, peak_to_peak=mx - 0.0))
    return df


def unscale_results(results, maximum):
    df = results.copy()
    # Un-scale results money info with manually set ranges for data
    # in params file.
    to_unscale = ['price', 'forecast', 'budget', 'cost',
                  'value', 'profit', 'balance']
    df[to_unscale] = unscale_columns(df[to_unscale], maximum)

    return df


def read_ohlc(filename: str, csv_dict: dict, **kwargs) -> DataFrame:
    """
    Reads a filename passed as CSV, renaming columns according to the
    dictionary passed.
    :param filename: the file with the ohlcv columns
    :param csv_dict: the dict with the names of the columns
    :return: the dataframe
    """
    filepath = file_exists(filename, dirname(realpath(__file__)))
    df = pd.read_csv(filepath, **kwargs)

    # Reorder and rename
    columns_order = [csv_dict[colname] for colname in csv_dict]
    df = df[columns_order]
    df.columns = csv_dict.keys()

    return df


def read_json(filename):
    """ Reads a JSON file, returns a dict. If file does not exists
    returns None """
    if os.path.exists(filename):
        with open(filename) as json_file:
            data = json.load(json_file)
        return data
    else:
        return None


def save_experiment(obj_name: str, folder: str, results: dict):
    """
    Creates a folder for the experiment and saves results. Results is a
    dictionary that will be saved as an opaque pickle. When the experiment will
    require to be loaded, the only parameter needed are the folder name.

    Args:
        obj_name (str): the name to be given to the pickle file to be saved. If
            a file already exists with that name, a file with same name and a
            extension will be generated.
        folder (str): a full path to the folder where the experiment is to be saved.
            If the folder does not exist it will be created.
        results (obj): the object to be saved as experiment. This is typically a
            dictionary with different items representing different parts of the
            experiment.

    Return:
        (str) The name under which the experiment has been saved
    """
    if not os.path.exists(folder):
        Path(folder).mkdir(parents=False, exist_ok=True)
    output = valid_output_name(obj_name, folder, extension="pickle")
    with open(output, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return output


def load_experiment(obj_name: str, folder: str):
    """
    Loads a pickle from a given folder name. It is not necessary to add the "pickle"
    extension to the experiment name.

    Parameters:
    -----------
    obj_name (str): The name of the object saved in pickle format that is to be loaded.
    folder (str): A full path where looking for the experiment object.

    Returns:
    --------
    An obj loaded from a pickle file.
    """
    if Path(obj_name).suffix == "" or Path(obj_name).suffix != "pickle":
        ext = ".pickle"
    else:
        ext = ''

    experiment = f"{str(Path(folder, obj_name))}{ext}"
    with open(experiment, 'rb') as h:
        return pickle.load(h)
