# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
import sys

import tqdm


class TqdmLoggingHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

# Idea borrowed from
# https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/utils/logging.py
def get_logger(name=None, level=logging.INFO) -> logging.Logger:
    """Returns a logger that is configured as:
    - by default INFO level or higher messages are logged out in STDOUT.
    - format includes file name, line number, etc.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    log_formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s] %(message)s"
    )
    handler_out: logging.StreamHandler = TqdmLoggingHandler(sys.stdout)
    handler_out.setFormatter(log_formatter)
    logger.addHandler(handler_out)

    return logger


# Delay evaluation of "logger" attribute so that capsys can capture the
# output of this logger.
def __getattr__(name):
    if name == "logger":
        return get_logger(name="MatterGen", level=logging.INFO)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
