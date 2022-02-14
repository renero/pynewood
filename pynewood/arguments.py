import argparse


class Arguments(object):
    args = None
    possible_actions = ['train', 'predict_training', 'predict',
                        'ensemble_predictions', 'ensemble']
    parser: argparse.ArgumentParser = None

    def __init__(self, *args):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument(
            '-c', '--config-file', nargs=1, type=str,
            help='Relative path to configuration file to be used (YAML).')
        self.parser.add_argument(
            '-f', '--file', nargs=1, type=str, required=True,
            help='Input OHLCV File to process')
        self.parser.add_argument(
            '-s', '--save', action='store_true',
            help='Save predictions, default OFF')
        self.parser.add_argument(
            '-e', '--epochs', nargs=1, type=int,
            help='Nr. of epochs for training. Default=1')
        self.parser.add_argument(
            '-o', '--output', nargs=1, type=str,
            help='Output filename to be used to save results (w/out extension)')
        self.parser.add_argument(
            '-p', '--plot', action='store_true',
            help='Plot a nice chart after training')
        self.parser.add_argument(
            '-d', '--debug', nargs=1, type=int,
            help='Debug level (0..4), default 3.')

        # Are there any args?
        if len(args[1]['args']) > 1:
            self.args = self.parser.parse_args(args[1]['args'][1:])
