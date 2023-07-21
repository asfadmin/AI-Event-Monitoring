"""
 Summary
 -------
 GUI plotting of data and interactive visualizations.

 Notes
 -----
 Created by Jason Herning, Andrew Player, and Robert Lawton.
"""


import matplotlib.pyplot as plt
import numpy as np


def show_dataset(masked: np.ndarray, wrapped: np.ndarray) -> None:
    """
    Plot the masked and wrapped arrays.

    Parameters
    -----------
    masked : np.ndarray
        The event-mask of the interferogram.
    wrapped : np.ndarray
        The wrapped interferogram.
    """

    _, [axs_masked, axs_wrapped] = plt.subplots(1, 2)

    axs_masked.set_title("Masked")
    axs_masked.imshow(masked, origin="lower", cmap="jet")

    axs_wrapped.set_title("Wrapped")
    axs_wrapped.imshow(wrapped, origin="lower", cmap="jet")

    plt.show()
