import matplotlib.pyplot as plt

from insar_eventnet.config import SYNTHETIC_DIR
from insar_eventnet.io import make_simulated_dataset, split_dataset, create_directories

name = "simulated_dataset"

amount = 2000
seed = 0
tile_size = 512
crop_size = 512
split = 0.2

create_directories()

name, count, dir_name, distribution, dataset_info = make_simulated_dataset(
    name, SYNTHETIC_DIR, amount, seed, tile_size, crop_size
)

dataset_path = SYNTHETIC_DIR.__str__() + "/" + dir_name

num_train, num_validation = split_dataset(dataset_path, split)