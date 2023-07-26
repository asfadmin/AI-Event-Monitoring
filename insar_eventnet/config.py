"""
 Summary
 -------
 Holds global variables.

 Notes
 -----
 Created by Jason Herning.
"""

import pathlib

# full Path to aiwater root
PROJECT_DIR = pathlib.Path.cwd().resolve()

# Path configurations for data directory
DATA_DIR = PROJECT_DIR / "data"

# Input data subdirectory path configs
INPUT_DIR = DATA_DIR / "input"
PRODUCTS_DIR = INPUT_DIR / "products"
AOI_DIR = INPUT_DIR / "aoi"

# TODO: test directory?
# Working data subdirectory path configs
WORKING_DIR = DATA_DIR / "working"
SYNTHETIC_DIR = WORKING_DIR / "synthetic"
REAL_DIR = WORKING_DIR / "real"

# Output data subdirectory path configs
OUTPUT_DIR = DATA_DIR / "output"
MODEL_DIR = OUTPUT_DIR / "models"
MASK_DIR = OUTPUT_DIR / "mask"
TENSORBOARD_DIR = OUTPUT_DIR / "tensorboard"

NETWORK_DEMS = 256
