import matplotlib.pyplot as plt

from insar_eventnet.inference import mask_image_path
from insar_eventnet.io import initialize, get_image_array

tile_size = 512
crop_size = 512

mask_model_path = "data/output/models/mask_model"
pres_model_path = "data/output/models/pres_model"
image_path = input("Image Path: ")  # Prompt user for input interferogram
image_name = image_path.split("/")[-1].split(".")[0]
output_path = f"masks_inferred/{image_name}_mask.tif"
image, gdal_dataset = get_image_array(image_path)

# The initialize function downloads the pretrained models
initialize()

mask, presence = mask_image_path(
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
