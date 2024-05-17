#!/usr/bin/env python
import logging
import os

from mmpie1 import Config
from mmpie1.utils import default_logger


def initial_mmpie1_setup():
    config = Config()
    new_dirs = []

    for dir_name in config['DIR_PATHS']:
        dir_path = config['DIR_PATHS'][dir_name]
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            new_dirs.append(dir_path)

    logger = default_logger(
        name=__name__,
        file_level=logging.INFO,
        file_name=os.path.join(config['DIR_PATHS']['LOGS'], 'mmpie1_setup_logs.txt'))
    for dir_ in new_dirs:
        logger.info(f'Created new directory: {dir_}')


if __name__ == '__main__':
    initial_mmpie1_setup()
