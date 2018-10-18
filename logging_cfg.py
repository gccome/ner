# -*- coding: utf-8 -*-

"""
    Author:   Daniel Yang
    Version:  1.0
    Date:     10/16/2017
    Project： PEH - Customer Insights Tool
    Module:   logging configuration
"""

import os
import logging.config
import json
import yaml


def setup_logging(default_path='logging.yaml', default_level=logging.INFO, env_key='LOG_CFG'):
    """
    Setup logging configuration
    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            # config = json.load(f)
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)