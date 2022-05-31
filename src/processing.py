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
            row_end = row + y_offset + tile_shape[0]
            col_end = col + x_offset + tile_shape[1]
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


def merge_offset(arr1: np.ndarray, arr2: np.ndarray, x_offset: int, y_offset: int) -> np.ndarray:

    """
    Merge two arrays which were both tiled from the same array, where one was tiled with offset.

    Parameters:
    -----------
    arr1 : np.ndarray
        This is the array that was tiled without offset.
    arr2 : np.ndarray
        This is the array that was tiled with offset.
    x_offset : int
        This is the amount that the tiling was offset in the x (column) direction.
    y_offset : int
        This is the amount that the tiling was offset in the y (row) direction.

    Returns:
    --------
    merged_arr : np.ndarray(shape=())
        This is the merged array.
    """

    rows = arr2.shape[0]
    cols = arr2.shape[1]
    arr1_part = arr1[0:(rows - y_offset), 0:(cols - x_offset)]
    arr2_part = arr2[y_offset:rows, x_offset:cols]
    merged_arr = (arr1_part + arr2_part) / 2
    return merged_arr


def simulate_unet_cropping(arr: np.ndarray, crop_shape: tuple):

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
            original_shape: Tuple[int, int],
            cropped:        bool = False,
            tiling_size:    int = 0
            ) -> np.ndarray:

    """
    Stich an array of 2-dimensional tiles into a single 2-dimensional array.

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

    print(original_shape)
    print(cropped)
    print(tiling_size)

    # These should be equal
    row_tile_size = len(arr[0, :, 0])
    col_tile_size = len(arr[0, 0, :])

    tile_size = row_tile_size

    rebuilt_arr = np.zeros((rows * tile_size, cols * tile_size))
    for i in range(rows):
        row = i * tile_size
        for j in range(cols):
            col = j * tile_size
            rebuilt_arr[row:row + tile_size, col:col + tile_size] = arr[(i * cols) + j]

    if cropped:
        crop_size = tile_size
        tile_size = tiling_size
        cl = (tile_size - crop_size) // 2
    else:
        crop_size = 0
        cl = 0

    (y, x) = original_shape

    x_padded = x + (tile_size - (x % tile_size)) + crop_size
    y_padded = y + (tile_size - (y % tile_size)) + crop_size

    start_row = ceil((y_padded - y) / 2)
    end_row = (start_row + y)
    start_col = ceil((x_padded - x) / 2)
    end_col = (start_col + x)

    rebuilt_arr = rebuilt_arr[start_row - cl:end_row - cl, start_col - cl:end_col - cl]

    return rebuilt_arr

# ------------------------------ #
# Experimental Merging Functions #
# ------------------------------ #


# Merge Tiles
def merge_tiles(tile_a: np.ndarray, tile_b: np.ndarray):

    """
    Experimental function that supports unidirectional_merge_tiles() -- Do not use right now.

    Parameters:
    -----------
    tile_a : np.ndarray
        tile from an array of tiles
    tile_b : np.ndarray
        tile from an array of tiles

    Returns:
    --------
    tile_a : np.ndarray
        tile_a merged with tile_b
    """

    delta = np.mean(tile_b - tile_a)
    jumpval = 2 * np.pi * np.round(delta / (2 * np.pi))
    tile_a += jumpval
    return tile_a


# Unidirectionally Merge Tiles
def unidirectional_merge_tiles(tiles: np.ndarray):

    """
    Experimental tile merging algoritm -- Do not use right now.

    Parameters:
    -----------
    tiles : np.ndarray
        3-dimensional array of tiles that needs to be merged

    Returns:
    --------
    tiles : np.ndarray
        2-dimensional array stiched from the merging of the tiles
    """

    rows = tiles.shape[0]
    cols = tiles.shape[1]
    for iw in range(cols):
        for j in range(rows):
            try:
                tiles[j, iw] = merge_tiles(tiles[j, iw], tiles[j - 1, iw])
            except IndexError:
                print("Nothing Above: {", iw, ", ", j, "}")
            try:
                tiles[j, iw] = merge_tiles(tiles[j, iw], tiles[j, iw - 1])
            except IndexError:
                print("Nothing to the Left: {", iw, ", ", j, "}")
    return tiles


def get_junctions(position: tuple):

    """
    Experimental function that supports tsrncp_merger() -- Do not use right now.

    Parameters:
    -----------
    position : tuple
        tuple representing the dimensions of an array of tiles

    Returns:
    --------
    final : list[list[tuple]]
        list of verticies representing the possible junctions of tiles
    """

    a = position[0]
    b = position[1]
    arr = [[(a, b), (a, b + 1)], [(a, b), (a + 1, b)], [(a, b), (a, b - 1)], [(a, b), (a - 1, b)]]
    final = arr
    i = 0
    for i in range(len(arr)):
        if arr[i][1][0] < 0 or arr[i][1][1] < 0:
            final.remove(arr[i])
    return final


def get_junctions_(dims: tuple):

    """
    Experimental function that supports tsrncp_merger() -- Do not use right now.

    Parameters:
    -----------
    dims : tuple
        tuple representing the dimensions of an array of tiles

    Returns:
    --------
    verts : list
        list of verticies representing the possible junctions of tiles
    """

    h = dims[0]
    w = dims[1]
    verts = [ ]
    for j in range(h):
        for i in range(w):
            if j + 1 < h:
                verts.append([(j, i), (j + 1, i)])
            if i + 1 < w:
                verts.append([(j, i), (j, i + 1)])
    return verts


def tsrncp_merger(tiles: np.ndarray):

    """
    Experimental merging algorithm -- Do not use right now.

    Parameters:
    -----------
    tiles : np.ndarray
        3-dimensional array of tiles which need to be merged

    Returns:
    --------
    tiles : np.ndarray
        2-dimensional array of merged tiles
    """

    dims = tiles.shape
    L = np.asarray(get_junctions_(dims))
    reliability = []
    for J in L:
        r = 1 / np.var(tiles[tuple(J[1])] - tiles[tuple(J[0])])
        reliability.append(r)
    sorted_indecies = np.argsort(reliability)
    rev_sorted_indecies = list(reversed(sorted_indecies))
    L_sorted = L[rev_sorted_indecies]
    groups = [ [ ] ]
    for junction in L_sorted:

        j0 = tuple(junction[0])
        j1 = tuple(junction[1])

        find_a = [x for x in groups if j0 in x]
        find_b = [x for x in groups if j1 in x]

        if len(find_a) == 0 and len(find_b) == 0:
            tiles[j0] = merge_tiles(tiles[j0], tiles[j1])
            groups.append([j0, j1])

        elif len(find_a) == 0 and len(find_b) > 0:
            tiles[j0] = merge_tiles(tiles[j0], tiles[j1])
            groups[groups.index(find_b[0])].append(j0)

        elif len(find_a) > 0 and len(find_b) == 0:
            tiles[j0] = merge_tiles(tiles[j1], tiles[j0])
            groups[groups.index(find_a[0])].append(j1)

        else:
            if groups[groups.index(find_a[0])] != groups[groups.index(find_b[0])]:
                index_a = groups.index(find_a[0])
                index_b = groups.index(find_b[0])
                if len(groups[index_a]) > len(groups[index_b]):
                    for tile in groups[index_b]:
                        tiles[j0] = merge_tiles(tiles[j0], tiles[tile])
                        groups[index_a].append(tile)
                    groups.pop(index_b)
                else:
                    for tile in groups[index_a]:
                        tiles[j1] = merge_tiles(tiles[j1], tiles[tile])
                        groups[index_b].append(tile)
                    groups.pop(index_a)

    return tiles
