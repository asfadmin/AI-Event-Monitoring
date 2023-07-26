"""
 Created By:   Andrew Player
 File Name:    test_processing.py
 Date Created: 07/26/2021
 Description:  unit test for image processing functions
"""

import numpy as np

from insar_eventnet.processing import pad, tile

test_arr_3x3 = np.zeros((3, 3))
test_arr_99x99 = np.zeros((99, 99))
test_arr_1031x925 = np.zeros((1031, 925))
test_arr_10031x9257 = np.zeros((10031, 9257))

padded_arr_3x3 = pad(test_arr_3x3, (2, 2))
padded_arr_99x99 = pad(test_arr_99x99, (20, 20))
padded_arr_10031x9257 = pad(test_arr_10031x9257, (256, 256))

(tiled_arr_3x3, _, _) = tile(test_arr_3x3, (2, 2), pad_value=-np.pi)
(tiled_arr_99x99, _, _) = tile(test_arr_99x99, (20, 20), pad_value=-np.pi)
(tiled_arr_1031x925, _, _) = tile(test_arr_1031x925, (256, 256), pad_value=-np.pi)
(tiled_arr_10031x9257, _, _) = tile(test_arr_10031x9257, (256, 256), pad_value=-np.pi)


def test_padding_shape():
    """
    Test that pad() is creating the correct minimumally divisible shapes.
    """

    assert padded_arr_3x3.shape == (4, 4)
    assert padded_arr_99x99.shape == (100, 100)
    assert padded_arr_10031x9257.shape == (10240, 9472)


def test_padding_value():
    """
    Test that pad() is placing the correct values in the correct places.
    """

    comp_arr_3x3 = np.full((4, 4), 0)
    comp_arr_3x3[:3, :3] = np.zeros((3, 3))

    comp_arr_99x99 = np.full((100, 100), 0)
    comp_arr_99x99[:99, :99] = np.zeros((99, 99))

    comp_arr_10031x9257 = np.full((10240, 9472), 0)
    comp_arr_10031x9257[:10031, :9257] = np.zeros((10031, 9257))

    assert np.array_equal(padded_arr_3x3, comp_arr_3x3)
    assert np.array_equal(padded_arr_99x99, comp_arr_99x99)
    assert np.array_equal(padded_arr_10031x9257, comp_arr_10031x9257)


def test_tiling_shape():
    """
    Test that tile() is creating arrays of the correct shape.
    """

    assert tiled_arr_3x3.shape == (4, 2, 2)
    assert tiled_arr_99x99.shape == (25, 20, 20)
    assert tiled_arr_1031x925.shape == (20, 256, 256)
    assert tiled_arr_10031x9257.shape == (1480, 256, 256)


def test_tiling_value():
    """
    Test that tile() is placing the correct values in the correct places.
    """

    comp_arr_3x3 = np.load("tests/data/3x3.npy")
    comp_arr_99x99 = np.load("tests/data/99x99.npy")
    comp_arr_1031x925 = np.load("tests/data/1031x925.npy")

    assert np.array_equal(tiled_arr_3x3, comp_arr_3x3)
    assert np.array_equal(tiled_arr_99x99, comp_arr_99x99)
    assert np.array_equal(tiled_arr_1031x925, comp_arr_1031x925)
