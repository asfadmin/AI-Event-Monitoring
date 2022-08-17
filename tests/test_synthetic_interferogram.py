"""
 Created By:   Jason Herning
 File Name:    test_synthetic_interferogram.py
 Date Created: 01-25-2021
 Description:  unit test for synthetic interferogram functions
"""

import numpy as np
import pytest
from astropy.modeling.models import Gaussian2D
from src.synthetic_interferogram import make_synthetic_interferogram, wrap_interferogram

# test gaussians
g2d1 = Gaussian2D(1.0, 150, 100, 20, 10, theta=0.5)
g2d2 = Gaussian2D(0.6,  50, 100, 20, 10, theta=0.5)
g2d3 = Gaussian2D(2.3,  23, 77 , 11,  5, theta=0.9)

# expected synthetic interferograms
y, x = np.mgrid[0:256, 0:256]
si_1g2d = (g2d1(x, y))
si_2g2d = (g2d1(x, y) + g2d2(x, y))
si_3g2d = (g2d1(x, y) + g2d2(x, y) + g2d3(x, y))

np.random.seed(20)
si_2g2d_noisy = si_2g2d + np.random.uniform(-10, 10, size=(256, 256))

# wrapped synthetic interferograms
si_1g2d_wrapped = np.angle(np.exp(1j*si_1g2d))
si_2g2d_wrapped = np.angle(np.exp(1j*si_2g2d))
si_3g2d_wrapped = np.angle(np.exp(1j*si_3g2d))
si_2g2d_noisy_wrapped = np.angle(np.exp(1j*si_2g2d_noisy))


@pytest.mark.parametrize("size, gaussians, expected", [(256, [g2d1], si_1g2d),
                                                       (256, [g2d1, g2d2], si_2g2d),
                                                       (256, [g2d1, g2d2], si_2g2d)])
def test_make_synthetic_interferogram(size, gaussians, expected):
    
    """
    Test that synthetic interferogram function works.
    """
    
    assert np.array_equal(make_synthetic_interferogram(size, *gaussians), expected)


@pytest.mark.parametrize(
                        "input_array   , expected",
                        [
                            (si_1g2d      , si_1g2d_wrapped),
                            (si_2g2d      , si_2g2d_wrapped),
                            (si_3g2d      , si_3g2d_wrapped),
                            (si_2g2d_noisy, si_2g2d_noisy_wrapped)
                        ])
def test_wrap_interferogram(input_array, expected):
    
    """
    Test interferogram wrapping function.
    """
    
    assert np.array_equal(wrap_interferogram(input_array), expected)
