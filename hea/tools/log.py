"""
Created on Thu Dec 17 17:23:38 2020

@author: ctetsass
"""

from sys import stdout

from loguru import logger as custom_logger


def create_logger():
    """Create custom logger."""

    custom_logger.remove()
    custom_logger.add(
        stdout,
        colorize=True,
        level="INFO",
        format="<light-cyan>{time:MM-DD-YYYY HH:mm:ss}</light-cyan> | <light-green>{level}</light-green>: "
               "<light-white>{message}</light-white>")

    custom_logger.add(
        'logs/errors.log',
        colorize=True,
        level="ERROR",
        rotation="200 MB",
        catch=True,
        format="<light-cyan>{time:MM-DD-YYYY HH:mm:ss}</light-cyan> | <light-cyan>{module}</light-cyan> - "
               "<light-cyan>{name}</light-cyan> | <light-red>{level}</light-red>: <light-white>{message}</light-white>")
    return custom_logger


logger = create_logger()
