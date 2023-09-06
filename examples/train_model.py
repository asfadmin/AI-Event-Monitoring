import matplotlib.pyplot as plt

from insar_eventnet import inference, io, training
from insar_eventnet.config import SYNTHETIC_DIR

# First generate a simulated dataset

dataset_name = "simulated_training_dataset"

amount = 2000
seed = 0  # if set to 0, the seed is randomized each run
tile_size = 512
crop_size = 512
split = 0.2  # Training/Testing split

io._create_directories()  # Initialize directory structure for training data

name, count, dir_name, distribution, dataset_info = io._make_simulated_dataset(
    dataset_name, SYNTHETIC_DIR, amount, seed, tile_size, crop_size
)

dataset_path = SYNTHETIC_DIR.__str__() + "/" + dir_name

num_train, num_validation = io._split_dataset(dataset_path, split)

# Then train a unet masking model on the simulated dataset

model_name = "mask_model_example"
model_type = "unet"
input_shape = tile_size
epochs = 15
filters = 64
batch_size = 1
learning_rate = 1e-4
use_wandb = False
using_aws = False
using_jupyter = False
logs_dir = ""

mask_model, mask_history = training.train(
    model_name,
    dataset_path,
    model_type,
    input_shape,
    epochs,
    filters,
    batch_size,
    learning_rate,
    use_wandb,
    using_aws,
    using_jupyter,
    logs_dir,
)

# Now, create a dataset of simulated events with masks from UNet
# See the model creation notebook for an explanation of why we don't train the binary
# prediction model on ground truth masks

name = "classfification_model_dataset"
mask_model_path = "data/output/models/checkpoints/" + model_name

amount = 1000
split = 0.1

name, count, dir_name, _, _ = io._make_simulated_dataset(
    name, SYNTHETIC_DIR, amount, seed, tile_size, crop_size, model_path=mask_model_path
)

dataset_path = SYNTHETIC_DIR.__str__() + "/" + dir_name

num_train, num_validation = io._split_dataset(dataset_path, split)

# Now train the binary classification model

model_name_bin = "pres_model_example"
model_type = "eventnet"
input_shape = crop_size
epochs = 5
filters = 64
batch_size = 1
learning_rate = 5e-3
use_wandb = False
using_aws = False
using_jupyter = False
logs_dir = ""

binary_model, binary_history = training.train(
    model_name_bin,
    dataset_path,
    model_type,
    input_shape,
    epochs,
    filters,
    batch_size,
    learning_rate,
    use_wandb,
    using_aws,
    using_jupyter,
    logs_dir,
)

# Now, we can run inference on these models!
mask_model_path = "data/output/models/mask_model_example"
pres_model_path = "data/output/models/pres_model_example"
image_path = input("Image Path: ")  # Prompt user for input interferogram
image_name = image_path.split("/")[-1].split(".")[0]
output_path = f"masks_inferred/{image_name}_mask.tif"
image, gdal_dataset = io._get_image_array(image_path)

mask, presence = inference.mask(
    mask_model_path=mask_model_path,
    pres_model_path=pres_model_path,
    image_path=image_path,
    tile_size=tile_size,
    crop_size=crop_size,
)

if presence > 0.7:
    print("Positive")
else:
    print("Negative")

_, [axs_wrapped, axs_mask] = plt.subplots(1, 2, sharex=True, sharey=True)

axs_wrapped.set_title("Wrapped")
axs_mask.set_title("Segmentation Mask")

axs_wrapped.imshow(image, origin="lower", cmap="jet")
axs_mask.imshow(mask, origin="lower", cmap="jet")

plt.show()
