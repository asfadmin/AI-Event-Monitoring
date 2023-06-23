import matplotlib.pyplot as plt

from insar_eventnet.config import SYNTHETIC_DIR
from insar_eventnet.sarsim import gen_simulated_deformation, gen_sim_noise

seed = 232323
tile_size = 512
event_type = "quake"

unwrapped_def, masked_def, wrapped_def, presence_def = gen_simulated_deformation(
    seed=seed, tile_size=tile_size, event_type=event_type
)
unwrapped_mix, masked_mix, wrapped_mix, presence_mix = gen_sim_noise(
    seed=seed, tile_size=tile_size
)

print(f"Deformation Presence: {presence_def}")
print(f"Mixed Noise Presence: {presence_mix}")

_, [axs_unwrapped_def, axs_wrapped_def, axs_mask_def] = plt.subplots(
    1, 3, sharex=True, sharey=True, tight_layout=True
)

_, [axs_unwrapped_mix, axs_wrapped_mix, axs_mask_mix] = plt.subplots(
    1, 3, sharex=True, sharey=True, tight_layout=True
)

axs_unwrapped_def.set_title("Deformation Event")
axs_unwrapped_mix.set_title("Atmospheric/Topographic Noise")

axs_unwrapped_def.imshow(unwrapped_def, origin="lower", cmap="jet")
axs_unwrapped_mix.imshow(unwrapped_mix, origin="lower", cmap="jet")
axs_wrapped_def.imshow(wrapped_def, origin="lower", cmap="jet")
axs_wrapped_mix.imshow(wrapped_mix, origin="lower", cmap="jet")
axs_mask_def.imshow(masked_def, origin="lower", cmap="jet", vmin=0.0, vmax=1.0)
axs_mask_mix.imshow(masked_mix, origin="lower", cmap="jet", vmin=0.0, vmax=1.0)

plt.show()
