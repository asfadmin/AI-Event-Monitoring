"""
 Created By:  Jason Herning
 File Name:   etc.py
 Description: misc functions
"""

import random
import sys


def new_seed():

    """
    Randomly generates a seed for use in synthetic interferogram dataset generation.

    Returns:
    --------
    n/a : float
        Randomly generated seed for use as a random seed.
    """

    seed_value = random.randrange(sys.maxsize)
    random.seed(seed_value)
    return random.randint(100000, 999999)
