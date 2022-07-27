"""
 Created By:   Andrew Player
 File Name:    processing.py
 Date Created: 01-25-2021
 Description:  Functions for the pre and post-processing of interferogram arrays
"""

from math import ceil
from typing import Tuple

import numpy as np


def tile(
    arr:        np.ndarray,
    tile_shape: Tuple[int, int],
    crop_size:  int   = 0,
    x_offset:   int   = 0,
    y_offset:   int   = 0,
    even_pad:   bool  = False,
    pad_value:  float = 0.0
) -> Tuple[np.ndarray, int, int]:

    """
    Tile a 2-dimensional array into an array of tiles of shape tile_shape.

    Parameters:
    -----------
    arr : np.ndarray
        The 2-dimensional array that should be tiled.
    tile_shape : tuple(int, int)
        The desired shape of the tiles: (row length, column length). This
        should match the input shape of the model.
    crop_size : int, Optional
        An extra amount of padding to maintain full array coverage when the model
        will crop the tiles later. This amount should be equal to the shape of the
        output tiles from the model.
    x_offset : int, Optional
        Offset the tiling process by this amount in the columns direction.
    y_offset : int, Optional
        Offset the tiling process by this amount in the row direction.
    even_pad : bool, Optional
        If True, the array will be padded symmetrically; else, it will be padded
        on the end of each dimension.
    pad_value : float, Optional
        The value to fill the padded areas of the array with.

    Returns:
    --------
    tiles : np.ndarray
        The array of tiles.
    num_rows : int
        The number of rows of tiles.
    num_cols : int
        The number of columns of tiles.
    """

    cropped = crop_size > 0
    rows, cols = len(arr[:, 0]), len(arr[0, :])

    if(cols % tile_shape[1] != 0 or rows % tile_shape[0] != 0):
        
        if x_offset > 0 or y_offset > 0:
            arr_offset = np.zeros((rows + y_offset, cols + x_offset))
            arr_offset[0:rows, 0:cols] = arr
            arr = arr_offset
        
        arr = pad(
            arr,
            tile_shape,
            even_pad=even_pad,
            extra_padding=(crop_size if cropped else 0),
            value=pad_value
        )
        
        rows, cols = len(arr[:, 0]), len(arr[0, :])

    # crop_offset_x => ((dims[1] + crop_size) // 2) - ((dims[1] - crop_size) // 2) = crop_size
    # crop_offset_y => ((dims[0] + crop_size) // 2) - ((dims[0] - crop_size) // 2) = crop_size

    if not cropped:
        num_rows  = rows // tile_shape[0] - ceil(y_offset / tile_shape[0])
        num_cols  = cols // tile_shape[1] - ceil(x_offset / tile_shape[1])
    else:
        num_rows  = ceil((rows - tile_shape[0]) / crop_size)  # = (h-th)/ch
        num_cols  = ceil((cols - tile_shape[1]) / crop_size)  # = (w-tw)/cw

    num_tiles = num_rows * num_cols

    # final_row_start = num_rows * crop_offset_y
    # final_col_start = num_cols * crop_offset_x

    tiles = np.zeros((num_tiles, tile_shape[0], tile_shape[1]))

    t_row = 0
    for i in range(num_rows):
        row = i * (crop_size if cropped else tile_shape[0])
        for j in range(num_cols):
            col = j * (crop_size if cropped else tile_shape[1])
            row_start = row + y_offset
            col_start = col + x_offset
            row_end   = row + y_offset + tile_shape[0]
            col_end   = col + x_offset + tile_shape[1]
            tiles[t_row, :, :] = arr[row_start:row_end, col_start:col_end]
            t_row += 1

    return tiles, num_rows, num_cols


def pad(
    arr:            np.ndarray,
    tile_shape:     Tuple[int, int],
    value:          float = 0.0,
    extra_padding:  int   = 0,
    even_pad:       bool  = False,
) -> np.ndarray:

    """
    Pad an array with a given value so that it can be tiled.

    Parameters:
    -----------
    arr : np.ndarray
        The array that should be padded.
    tile_shape : tuple(int, int)
        A tuple representing the shape of the tiles (rows, columns) to pad for.
    value : float, Optional
        The value that should fill the padded area.
    extra_pad : int, Optional
        An extra amount of padding to add to each dimension.
    even_padding : bool, Optional
        If True, the array will be padded symmetrically; else, it will be padded
        on the end of each dimension.

    Returns:
    --------
    arr_padded : np.ndarray
        The padded array.
    """

    y, x = len(arr[:, 0]), len(arr[0, :])
    x_padded = x + (tile_shape[0] - (x % tile_shape[0])) + extra_padding
    y_padded = y + (tile_shape[1] - (y % tile_shape[1])) + extra_padding
    arr_padded = np.full((y_padded, x_padded), value)
    if even_pad:
        start_row = ceil((y_padded - y) / 2)
        end_row = start_row + y
        start_col = ceil((x_padded - x) / 2)
        end_col = start_col + x
        arr_padded[start_row:end_row, start_col:end_col] = arr
    else:
        arr_padded[:y, :x] = arr
    return arr_padded


def simulate_unet_cropping(
    arr: np.ndarray,
    crop_shape: tuple
) -> np.ndarray:

    """
    Symmetrically crop the inputed array.

    Parameters:
    -----------
    arr : np.ndarray
        The 2-dimensional interferogram array to be cropped.
    crop_shape : tuple
        The length of the rows and columns after being cropped, respectively. If the crop_shape
        in a direction does not divide 2, the extra value will be placed on the 0 side. This should
        match the models output shape.

    Returns:
    --------
    cropped_arr : np.ndarray
        The cropped interferogram array.
    """

    startx = arr.shape[1] // 2 - ceil(crop_shape[1] / 2)
    starty = arr.shape[0] // 2 - ceil(crop_shape[0] / 2)
    cropped_arr = arr[starty:starty + crop_shape[0], startx:startx + crop_shape[1]]

    return cropped_arr


def tiles_to_image(
    arr:            np.ndarray,
    rows:           int,
    cols:           int,
    original_shape: Tuple[int, int]
) -> np.ndarray:

    """
    Stich an array of 2-dimensional tiles into a single 2-dimensional array.
s
    Parameters:
    -----------
    arr : np.ndarray(shape=(rows*cols, tile_size, tile_size))
        The array of tiles to be stiched together.
    rows : int
        The number of tiles that should go in the row direction.
    cols : int
        The number of tiles that should go in the column direction.
    original_shape : Tuple[int, int]
        The shape of the interferogram before any processing.
    cropped : bool
        Set to True if the model has cropping, else leave it as False.
    tiling_size : int
        If the model was cropped, set this to the original tile size, else
        leave this false.

    Returns:
    --------
    rebuilt_arr : np.ndarray(shape=(rows*tile_size, cols*tile_size))
        The stiched 2-dimensional array.
    """

    row_tile_size = len(arr[0, :, 0])
    col_tile_size = len(arr[0, 0, :])

    assert row_tile_size == col_tile_size

    tile_size = row_tile_size

    rebuilt_arr = np.zeros((rows * tile_size, cols * tile_size))
    for i in range(rows):
        row = i * tile_size
        for j in range(cols):
            col = j * tile_size
            rebuilt_arr[row:row + tile_size, col:col + tile_size] = arr[(i * cols) + j]

    (y, x) = original_shape

    x_padded = x + (tile_size - (x % tile_size))
    y_padded = y + (tile_size - (y % tile_size))

    start_row = ceil((y_padded - y) / 2)
    end_row = (start_row + y)
    start_col = ceil((x_padded - x) / 2)
    end_col = (start_col + x)

    rebuilt_arr = rebuilt_arr[start_row:end_row, start_col:end_col]

    return rebuilt_arr