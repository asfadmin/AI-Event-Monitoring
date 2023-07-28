"""
 Summary
 -------
 Functions for simulated-deformation interferogram generation for training datasets.

 References
 ----------
 Functions taken from https://github.com/matthew-gaddes/SyInterferoPy.

 -----
 Created by Andrew Player.
"""


import time
import random

import numpy as np
import numpy.ma as ma
from geopy import distance

from insar_eventnet import synthetic_interferogram
import numpy as np


class Okada:

    """
    Class that models surface deformation.

    Based off of:

    https://github.com/matthew-gaddes/SyInterferoPy

    and

    Okada, Surface deformation due to shear and tensile faults in a half-space,
    Bulletin of the Seismological Society of America (1985): 1135-1154
    """

    def __init__(self, source, source_xy_m, tile_size, **kwargs):
        np.seterr(divide="ignore")

        self.source_type = source
        self.source_x = source_xy_m[0]
        self.source_y = source_xy_m[1]
        self.tile_size = tile_size
        self.params = kwargs

        self.gen_coordinates()

        self.los_vector = np.array([[0.38213591], [-0.08150437], [0.92050485]])

        # Lames Constants and Poisson Ratio
        self.lames_mu = 2.3e10  # μ
        self.lames_lambda = 2.3e10  # λ
        self.nu = self.lames_lambda / (
            2 * (self.lames_lambda + self.lames_mu)
        )  # Poisson Ratio: ν

        self.length = kwargs["length"]
        self.strike = np.deg2rad(kwargs["strike"])

        # δ
        self.dip = np.deg2rad(kwargs["dip"])

        # Parameters Depending on Type of Event
        if source == "quake":
            self.opening = 0
            self.slip = kwargs["slip"]
            self.rake = np.deg2rad(kwargs["rake"])
            self.width = kwargs["bottom_depth"] - kwargs["top_depth"]
            self.depth = np.mean((kwargs["bottom_depth"], kwargs["top_depth"]))

        elif source == "dyke":
            self.opening = kwargs["opening"]
            self.slip = 0
            self.rake = np.deg2rad(0)
            self.width = kwargs["bottom_depth"] - kwargs["top_depth"]
            self.depth = np.mean((kwargs["bottom_depth"], kwargs["top_depth"]))

        elif source == "sill":
            self.opening = kwargs["opening"]
            self.slip = 0
            self.rake = np.deg2rad(0)
            self.depth = kwargs["depth"]
            self.width = kwargs["width"]

        else:
            raise Exception(
                f"'Source' must be either 'quake', 'dyke', or 'sill', but is set to {source}.  Exiting."
            )

        # Components in North and East Directions
        self.east = (
            self.grid_x
            - self.source_x
            + np.cos(self.strike) * np.cos(self.dip) * (self.width / 2)
        )
        self.north = (
            self.grid_y
            - self.source_y
            - np.sin(self.strike) * np.cos(self.dip) * (self.width / 2)
        )

        # ξ is okada_x
        self.okada_x = (
            np.cos(self.strike) * self.north
            + np.sin(self.strike) * self.east
            + (self.length / 2)
        )
        self.okada_y = (
            np.sin(self.strike) * self.north
            - np.cos(self.strike) * self.east
            + (np.cos(self.dip) * self.width)
        )

        self.d = self.depth + np.sin(self.dip) * (self.width / 2)
        self.q = (self.okada_y * np.sin(self.dip)) - (self.d * np.cos(self.dip))

        # η
        self.eta = (self.okada_y * np.cos(self.dip)) + (self.d * np.sin(self.dip))

        self.compute_I_components()

        self.U1 = np.cos(self.rake) * self.slip
        self.U2 = np.sin(self.rake) * self.slip
        self.U3 = self.opening

        self.compute_displacement()

    def gen_coordinates(self):
        x_axis, y_axis = np.meshgrid(  # Coordinate Axes
            np.arange(0, self.tile_size)
            * 90,  # ((45990 * (self.tile_size / 512)) / (self.tile_size - 1)), # 90m/pixel in x direction
            np.arange(0, self.tile_size)
            * 90,  # ((45990 * (self.tile_size / 512)) / (self.tile_size - 1)) # 90m/pixel in y direction
        )

        y_axis = np.flipud(y_axis)

        ij_bases = np.vstack(  # Coordinate System Basis i and j
            (np.ravel(x_axis)[np.newaxis], np.ravel(y_axis)[np.newaxis])
        )

        ijk_bases = np.vstack(  # Coordinate System Basis i, j, and k
            (ij_bases, np.zeros((1, ij_bases.shape[1])))
        )

        self.x_axis_shape = x_axis.shape
        self.y_axis_shape = y_axis.shape
        self.grid_x = ijk_bases[0,]
        self.grid_y = ijk_bases[1, :]

    def chinnery(self, f):
        """
        Method of combining the different components of displacement.
        """

        return (
            f(0, 0) - f(self.width, 0) - f(0, self.length) + f(self.width, self.length)
        )

    def compute_displacement(self):
        """
        Compute the displacements in all directions from all of the sources and combine them.
        """

        if self.source_type == "quake":
            U_1 = (self.U1 / (2 * np.pi)) * self.chinnery(self.strike_slip_displacement)
            U_2 = (self.U2 / (2 * np.pi)) * self.chinnery(self.dip_slip_displacement)
        else:
            U_1 = 0
            U_2 = 0

        if self.source_type == "sill" or self.source_type == "dyke":
            U_3 = (self.U3 / (2 * np.pi)) * self.chinnery(self.tensile_displacement)
        else:
            U_3 = 0

        okada_array = -U_1 - U_2 + U_3

        displacement_array = np.zeros((3, self.tile_size * self.tile_size))

        displacement_array[0] = (
            np.sin(self.strike) * okada_array[0] - np.cos(self.strike) * okada_array[1]
        )
        displacement_array[1] = (
            np.cos(self.strike) * okada_array[0] + np.sin(self.strike) * okada_array[1]
        )
        displacement_array[2] = okada_array[2]

        self.displacement = displacement_array

        shapes = (self.x_axis_shape[0], self.x_axis_shape[1])
        x_grid = (
            np.reshape(
                self.displacement[0,],
                shapes,
            )
            * self.los_vector[0, 0]
        )
        y_grid = (
            np.reshape(
                self.displacement[1,],
                shapes,
            )
            * self.los_vector[1, 0]
        )
        z_grid = (
            np.reshape(
                self.displacement[2,],
                shapes,
            )
            * self.los_vector[2, 0]
        )

        self.los_displacement = x_grid + y_grid + z_grid

    def update_params_WL(self, W, L):
        xi = self.okada_x - L
        eta = self.eta - W

        self.R = np.sqrt(np.square(xi) + np.square(eta) + np.square(self.q))
        self.X = np.sqrt(np.square(xi) + np.square(self.q))
        self.y_tilda = (eta * np.cos(self.dip)) + (self.q * np.sin(self.dip))
        self.d_tilda = (eta * np.sin(self.dip)) - (self.q * np.cos(self.dip))

        return xi, eta

    def compute_I_1(self, W, L):
        xi = self.okada_x - L

        if W != 0 and L != 0:
            I_5 = self.I_5_WL
        elif W != 0:
            I_5 = self.I_5_W
        elif L != 0:
            I_5 = self.I_5_L
        else:
            I_5 = self.I_5

        if np.cos(self.dip) > 10e-8:
            return (1 - 2 * self.nu) * (
                -xi / (np.cos(self.dip) * (self.R + self.d_tilda))
            ) - np.tan(self.dip) * I_5
        else:
            return -((1 - 2 * self.nu) / 2) * (
                (xi * self.q) / np.square((self.R + self.d_tilda))
            )

    def compute_I_2(self, W, L):
        eta = self.eta - W

        if W != 0 and L != 0:
            I_3 = self.I_3_WL
        elif W != 0:
            I_3 = self.I_3_W
        elif L != 0:
            I_3 = self.I_3_L
        else:
            I_3 = self.I_3

        return (1 - 2 * self.nu) * -np.log(self.R + eta) - I_3

    def compute_I_3(self, W, L):
        eta = self.eta - W

        if W != 0 and L != 0:
            I_4 = self.I_4_WL
        elif W != 0:
            I_4 = self.I_4_W
        elif L != 0:
            I_4 = self.I_4_L
        else:
            I_4 = self.I_4

        if np.cos(self.dip) > 10e-8:
            return (1 - 2 * self.nu) * (
                (1 / (np.cos(self.dip))) * (self.y_tilda / (self.R + self.d_tilda))
                - np.log(self.R + eta)
            ) + np.tan(self.dip) * I_4
        else:
            return ((1 - 2 * self.nu) / 2) * (
                (eta / (self.R + self.d_tilda))
                + ((self.y_tilda * self.q) / np.square(self.R + self.d_tilda))
                - np.log(self.R + eta)
            )

    def compute_I_4(self, W):
        eta = self.eta - W

        if np.cos(self.dip) > 10e-8:
            return (
                (1 - 2 * self.nu)
                * (1 / np.cos(self.dip))
                * (
                    np.log(self.R + self.d_tilda)
                    - np.sin(self.dip) * np.log(self.R + eta)
                )
            )
        else:
            return -(1 - 2 * self.nu) * (self.q / (self.R + self.d_tilda))

    def compute_I_5(self, W, L):
        xi = self.okada_x - L
        eta = self.eta - W

        if np.cos(self.dip) > 10e-8:
            a = eta * (self.X + (self.q * np.cos(self.dip))) + (
                self.X * (self.R + self.X) * np.sin(self.dip)
            )
            b = xi * (self.R + self.X) * np.cos(self.dip)
            return (1 - 2 * self.nu) * 2 * np.arctan(a / b) / np.cos(self.dip)
        else:
            return -(1 - 2 * self.nu) * (
                (xi * np.sin(self.dip)) / (self.R + self.d_tilda)
            )

    def compute_I_components(self):
        """
        Precompute all necessary I components to avoid doing it more than once.
        """

        self.update_params_WL(0, 0)

        self.I_5 = self.compute_I_5(0, 0)
        self.I_4 = self.compute_I_4(0, 0)
        self.I_3 = self.compute_I_3(0, 0)
        self.I_2 = self.compute_I_2(0, 0)
        self.I_1 = self.compute_I_1(0, 0)

        self.update_params_WL(self.width, 0)

        self.I_5_W = self.compute_I_5(self.width, 0)
        self.I_4_W = self.compute_I_4(self.width, 0)
        self.I_3_W = self.compute_I_3(self.width, 0)
        self.I_2_W = self.compute_I_2(self.width, 0)
        self.I_1_W = self.compute_I_1(self.width, 0)

        self.update_params_WL(0, self.length)

        self.I_5_L = self.compute_I_5(0, self.length)
        self.I_4_L = self.compute_I_4(0, self.length)
        self.I_3_L = self.compute_I_3(0, self.length)
        self.I_2_L = self.compute_I_2(0, self.length)
        self.I_1_L = self.compute_I_1(0, self.length)

        self.update_params_WL(self.width, self.length)

        self.I_5_WL = self.compute_I_5(self.width, self.length)
        self.I_4_WL = self.compute_I_4(self.width, self.length)
        self.I_3_WL = self.compute_I_3(self.width, self.length)
        self.I_2_WL = self.compute_I_2(self.width, self.length)
        self.I_1_WL = self.compute_I_1(self.width, self.length)

    def get_Is(self, W, L):
        """
        Get the I components corresponding to W and L
        """

        if W != 0 and L != 0:
            I_1 = self.I_1_WL
            I_2 = self.I_2_WL
            I_3 = self.I_3_WL
            I_4 = self.I_4_WL
            I_5 = self.I_5_WL
        elif W != 0:
            I_1 = self.I_1_W
            I_2 = self.I_2_W
            I_3 = self.I_3_W
            I_4 = self.I_4_W
            I_5 = self.I_5_W
        elif L != 0:
            I_1 = self.I_1_L
            I_2 = self.I_2_L
            I_3 = self.I_3_L
            I_4 = self.I_4_L
            I_5 = self.I_5_L
        else:
            I_1 = self.I_1
            I_2 = self.I_2
            I_3 = self.I_3
            I_4 = self.I_4
            I_5 = self.I_5

        return I_1, I_2, I_3, I_4, I_5

    def strike_slip_displacement(self, W, L):
        xi, eta = self.update_params_WL(W, L)

        I_1, I_2, _, I_4, _ = self.get_Is(W, L)

        q_re = self.q / (self.R + eta)
        q_rre = (1 / self.R) * q_re
        arctan = np.arctan((xi * eta) / (self.q * self.R))

        x_direction = (xi * q_rre) + arctan + (I_1 * np.sin(self.dip))
        y_direction = (
            (self.y_tilda * q_rre)
            + (np.cos(self.dip) * q_re)
            + (I_2 * np.sin(self.dip))
        )
        z_direction = (
            (self.d_tilda * q_rre)
            + (np.sin(self.dip) * q_re)
            + (I_4 * np.sin(self.dip))
        )

        return np.asarray([x_direction, y_direction, z_direction])

    def dip_slip_displacement(self, W, L):
        xi, eta = self.update_params_WL(W, L)

        I_1, _, I_3, _, I_5 = self.get_Is(W, L)

        sc_dip = np.sin(self.dip) * np.cos(self.dip)
        q_rrx = self.q / (self.R * (self.R + xi))
        arctan = np.arctan((xi * eta) / (self.q * self.R))

        x_direction = (self.q / self.R) - I_3 * sc_dip
        y_direction = (
            (self.y_tilda * q_rrx) + (np.cos(self.dip) * arctan) - (I_1 * sc_dip)
        )
        z_direction = (
            (self.d_tilda * q_rrx) + (np.sin(self.dip) * arctan) - (I_5 * sc_dip)
        )

        return np.asarray([x_direction, y_direction, z_direction])

    def tensile_displacement(self, W, L):
        xi, eta = self.update_params_WL(W, L)

        I_1, _, I_3, _, I_5 = self.get_Is(W, L)

        s_d = np.square(np.sin(self.dip))
        q_rre = self.q / (self.R * (self.R + eta))
        q_rrx = self.q / (self.R * (self.R + xi))
        arctan = np.arctan((xi * eta) / (self.q * self.R))
        q_rre_arctan = (xi * q_rre) - arctan

        x_direction = (self.q * q_rre) - (I_3 * s_d)
        y_direction = (
            -(self.d_tilda * q_rrx) - (np.sin(self.dip) * q_rre_arctan) - (I_1 * s_d)
        )
        z_direction = (
            (self.y_tilda * q_rrx) + (np.cos(self.dip) * q_rre_arctan) - (I_5 * s_d)
        )

        return np.asarray([x_direction, y_direction, z_direction])


def atmosphere_turb(n_atms, lons_mg, lats_mg, mean_m=0.02):
    """
    A function to create synthetic turbulent atmospheres based on the  methods in Lohman Simons 2005, or using Andy Hooper and Lin Shen's fft method.
    Note that due to memory issues, when using the covariance (Lohman) method, largers ones are made by interpolateing smaller ones.
    Can return atmsopheres for an individual acquisition, or as the difference of two (as per an interferogram).  Units are in metres.

    Parameters
    ----------
    n_atms : int
        number of atmospheres to generate
    lons_mg : rank 2 array
        longitudes of the bottom left corner of each pixel.
    lats_mg : rank 2 array
        latitudes of the bottom left corner of each pixel.
    method : string
        'fft' or 'cov'.  Cov for the Lohmans Simons (sp?) method, fft for Andy
        Hooper/Lin Shen's fft method (which is much faster).  Currently no way to set
        length scale using fft method.
    mean_m : float
        average max or min value of atmospheres that are created.  e.g. if 3 atmospheres
        have max values of 0.02m, 0.03m, and 0.04m, their mean would be 0.03cm.
    water_mask : rank 2 array
        If supplied, this is applied to the atmospheres generated, convering them to masked arrays.
    difference : boolean
        If difference, two atmospheres are generated and subtracted from each other to
        make a single atmosphere.
    verbose : boolean
        Controls info printed to screen when running.
    cov_Lc : float
        length scale of correlation, in metres.  If smaller, noise is patchier, and if
        larger, smoother.
    cov_interpolate_threshold : int
        if n_pixs is greater than this, images will be generated at size so that the
        total number of pixels doesn't exceed this. e.g. if set to 1e4 (10000, the
        default) and images are 120*120, they will be generated at 100*100 then
        upsampled to 120*120.


    Returns
    -------
    ph_turbs : r3 array
        n_atms x n_pixs x n_pixs, UNITS ARE M.  Note that if a water_mask is provided, this is applied and a masked array is returned.
    """

    def lon_lat_to_ijk(lons_mg, lats_mg):
        """
        Given a meshgrid of the lons and lats of the lower left corner of each pixel,
        find their distances (in metres) from the lower left corner.

        Parameters
        -------
        lons_mg : rank 2 array
            longitudes of the lower left of each pixel.
        lats_mg : rank 2 array
            latitudes of the lower left of each pixel.

        Returns
        -------
        ijk : rank 2 array
            3x lots.  The distance of each pixel from the lower left corner of the image in metres.
        pixel_spacing : dict
            size of each pixel (ie also the spacing between them) in 'x' and 'y' direction.
        """
        ny, nx = lons_mg.shape

        pixel_spacing = {
            "x": distance.distance(
                (lats_mg[-1, 0], lons_mg[-1, 0]), (lats_mg[-1, 0], lons_mg[-1, 1])
            ).meters,
            "y": distance.distance(
                (lats_mg[-1, 0], lons_mg[-1, 0]), (lats_mg[-2, 0], lons_mg[-1, 0])
            ).meters,
        }

        X, Y = np.meshgrid(
            pixel_spacing["x"] * np.arange(0, nx), pixel_spacing["y"] * np.arange(0, ny)
        )
        Y = np.flipud(Y)

        ij = np.vstack((np.ravel(X)[np.newaxis], np.ravel(Y)[np.newaxis]))
        ijk = np.vstack((ij, np.zeros((1, ij.shape[1]))))

        return ijk, pixel_spacing

    def generate_correlated_noise_fft(nx, ny, std_long, sp):
        """A function to create synthetic turbulent troposphere delay using an FFT approach.
        The power of the turbulence is tuned by the weather model at the longer wavelengths.

        Parameters
        -------
        nx : int
            width of troposphere
        Ny : int
            length of troposphere
        std_long : float
            standard deviation of the weather model at the longer wavelengths. Default = ?
        sp : int
            pixel spacing in km

        Returns
        -------
        APS : float
            2D array, Ny * nx, units are m.
        """

        np.seterr(divide="ignore")

        cut_off_freq = 1 / 50

        x, y = np.arange(0, int(nx / 2)), np.arange(0, int(ny / 2))

        freq_x = np.divide(x, nx * sp)
        freq_y = np.divide(y, ny * sp)

        Y, X = np.meshgrid(freq_x, freq_y)

        freq = np.sqrt((X * X + Y * Y) / 2)

        ix = freq < 2 / 3
        log_power = np.log10(freq) * -11 / 3
        log_power[ix] = np.log10(freq[ix]) * -8 / 3 - np.log10(2 / 3)

        ix = freq < cut_off_freq
        bin_power = np.power(10, log_power)
        bin_power[ix] = 0

        APS_power = np.zeros((ny, nx))

        APS_power[0 : int(ny / 2), 0 : int(nx / 2)] = bin_power
        APS_power[0 : int(ny / 2), int(np.ceil(nx / 2)) :] = np.fliplr(bin_power)
        APS_power[int(np.ceil(ny / 2)) :, 0 : int(nx / 2)] = np.flipud(bin_power)
        APS_power[int(np.ceil(ny / 2)) :, int(np.ceil(nx / 2)) :] = np.fliplr(
            np.flipud(bin_power)
        )

        APS_filt = np.sqrt(APS_power)

        x = np.random.randn(ny, nx)
        y_tmp = np.fft.fft2(x)
        y_tmp2 = np.multiply(y_tmp, APS_filt)
        y = np.fft.ifft2(y_tmp2)

        APS = np.real(y)
        return (APS / np.std(APS)) * std_long * 0.01

    def rescale_atmosphere(atm, atm_mean=0.02, atm_sigma=0.005):
        """
        a function to rescale a 2d atmosphere with any scale to a mean centered
        one with a min and max value drawn from a normal distribution.

        Parameters
        ----------
        atm : rank 2 array
            a single atmosphere.
        atm_mean : float
            average max or min value of atmospheres that are created, in metres.  e.g. if 3 atmospheres have max values of 0.02m, 0.03m, and 0.04m, their mean would be 0.03m
        atm_sigma : float
            standard deviation of Gaussian distribution used to generate atmosphere strengths.

        Returns
        -------
        atm : rank 2 array
            a single atmosphere, rescaled to have a maximum signal of around that set by mean_m
        """

        atm -= np.mean(atm)

        atm_strength = (atm_sigma * np.random.randn(1)) + atm_mean

        if np.abs(np.min(atm)) > np.abs(np.max(atm)):
            atm *= atm_strength / np.abs(np.min(atm))
        else:
            atm *= atm_strength / np.max(atm)

        return atm

    ny, nx = lons_mg.shape

    ph_turbs = np.zeros((n_atms, ny, nx))
    _, pixel_spacing = lon_lat_to_ijk(lons_mg, lats_mg)

    for i in range(n_atms):
        ph_turbs[i, :, :] = generate_correlated_noise_fft(
            nx,
            ny,
            std_long=1,
            sp=0.0009 * np.mean((pixel_spacing["x"], pixel_spacing["y"])),
        )

    ph_turbs_m = np.zeros(ph_turbs.shape)
    for atm_n, atm in enumerate(ph_turbs):
        ph_turbs_m[atm_n,] = rescale_atmosphere(atm, mean_m)

    return ph_turbs_m[:, : lons_mg.shape[0], : lons_mg.shape[1]]


def gen_fake_topo(size: int = 512, alt_scale_max: int = 500):
    """
    Generate fake topography (a dem in meters) for generating simulated atmospheric topographic error.

    Parameters
    -----------
    size : int
        The size n for the (n, n) dimension array that is returned.
    alt_scale_min : int
        The minimum altitude scaling value for the generated perlin noise.
    alt_scale_max : int
        The maximum altitude scaling value for the generated perlin noise.

    Returns
    --------
    dem : np.ndarray
        The array that is meant to be used as a simulated dem with values in meters.
    """

    dem = np.zeros((size, size))
    dem = synthetic_interferogram.generate_perlin(dem.shape[0]) * alt_scale_max

    neg_indices = dem < np.max(dem) / 1.75

    dem[neg_indices] = 0

    non_zero_indices = dem > 0

    return dem[non_zero_indices] - np.min(dem[non_zero_indices])


def atm_topo_simulate(
    dem_m: np.ndarray,
    strength_mean: float = 56.0,
    strength_var: float = 2.0,
):
    """`
    Generate simulated topographic atmospheric error.

    Parameters
    -----------
    dem_m : np.ndarray
        An array containing either real dem values (in meters) or simulated ones.
    strength_mean : float
        The mean strength/magnitude of the error.
    strength_var : float
        The maximum variation from strength_mean of the magnitude of the error.
    difference : bool
        Whether the error should come from the difference of 2 aquisitions or just 1.

    Returns
    --------
    ph_turb : np.ndarray
        The array containing the turbulent atmospheric error.
    """

    # Sentinel-1 Wavelength in meters
    s1_lambda = 0.0547
    dem_km = 0.01 * dem_m

    ph_topo_aq1 = (
        strength_mean + strength_var * np.random.randn(1)
    ) * dem_km  # this is the delay for one acquisition
    ph_topo_aq2 = (
        strength_mean + strength_var * np.random.randn(1)
    ) * dem_km  # and for another
    ph_topo_m = (
        (ph_topo_aq1 - ph_topo_aq2) / (4 * np.pi)
    ) * s1_lambda  # interferogram is the difference, converted from rad to meters

    if np.max(ph_topo_m) < 0:
        ph_topo_m -= np.max(ph_topo_m)
    else:
        ph_topo_m -= np.min(ph_topo_m)

    ph_topo_m = ma.array(ph_topo_m, mask=ma.getmask(dem_m))
    ph_topo_m -= ma.mean(ph_topo_m)

    return ph_topo_m


def aps_simulate(size: int = 512):
    """
    Generate simulated turbulent atmospheric error.

    Parameters
    -----------
    size : int
        The size n for the (n, n) dimension array that is returned.

    Returns
    --------
    ph_turb : np.ndarray
        The array containing the turbulent atmospheric error.
    """

    pixel_size_degs = 1 / 3600

    scaled_size = pixel_size_degs * size

    lons = np.arange(0.0, 0.0 + scaled_size, pixel_size_degs)
    lats = np.arange(0.0, 0.0 + scaled_size, pixel_size_degs)

    lons_mg = np.repeat(lons[np.newaxis, :], len(lats), axis=0)
    lats_mg = np.repeat(lats[::-1, np.newaxis], len(lons), axis=1)

    ph_turb = atmosphere_turb(1, lons_mg, lats_mg, mean_m=0.02)

    return ph_turb[0,]


def coherence_mask_simulate(size: int = 512, threshold: float = 0.3):
    """
    Generate simulated incoherence to be masked out.

    Parameters
    -----------
    size : int
        The size n for the (n, n) dimension array that is returned.
    threshold : float
        The maximum value of coherence to be masked to zeros.

    Returns
    --------
    mask_coh : np.ndarray
        The masked coherence array.
    """

    pixel_size_degs = 1 / 3600

    lons = np.arange(0.0, 0.0 + (pixel_size_degs * size), pixel_size_degs)
    lats = np.arange(0.0, 0.0 + (pixel_size_degs * size), pixel_size_degs)
    lons_mg = np.repeat(lons[np.newaxis, :], len(lats), axis=0)
    lats_mg = np.repeat(lats[::-1, np.newaxis], len(lons), axis=1)

    mask_coh_values = atmosphere_turb(1, lons_mg, lats_mg, mean_m=0.01)

    mask_coh_values = (mask_coh_values - np.min(mask_coh_values)) / np.max(
        mask_coh_values - np.min(mask_coh_values)
    )
    return np.where(
        mask_coh_values > threshold, np.ones(lons_mg.shape), np.zeros(lons_mg.shape)
    )


def gen_simulated_deformation(
    seed: int = 0,
    tile_size: int = 512,
    log: bool = False,
    atmosphere_scalar: float = 100 * np.pi,
    amplitude_scalar: float = 1000 * np.pi,
    event_type: str = "quake",
    **kwargs,
):
    """
    Generate a wrapped interferogram along with an event-mask from simulated deformation

    Parameters
    -----------
    seed : int, Optional
        A seed for the random functions. For the same seed, with all other values the same
        as well, the interferogram generation will have the same results. If left at 0,
        the results will be different every time.
    tile_size : int, Optional
        The desired dimensional size of the interferogram pairs. This should match
        the input shape of the model.
    log : bool, Optional
        If true, the function will log various relevant values in the console.
    atmosphere_scalar : float, Optional
        Scale factor for the intensity of atmospheric noise.
    amplitude_scalar : float, Optional
        Scale factor for the deformation.
    event_type : str, Optional
        The type of deformation event. Can be quake, sill, or dyke.

    Returns
    --------
    masked_grid : np.ndarray(shape=(tile_size, tile_size))
        An array representing a mask over the simulated deformation which simulates masking an event.
    wrapped_grid : np.ndarray(shape=(tile_size, tile_size)
        The wrapped interferogram.
    presence : [1] or [0]
        [1] if the image contains an event else [0]
    """

    if seed != 0:
        np.random.seed(seed)

    presence = np.asarray([1])

    masked_grid = np.zeros((tile_size, tile_size))

    random_nums = np.random.rand(13)

    if not kwargs:
        axes_max = (tile_size) * 90

        source_x = axes_max // ((random_nums[0] * 10) + 1)
        source_y = axes_max // ((random_nums[1] * 10) + 1)

        top_depth = 6000 + 5000 * random_nums[3]

        if event_type == "quake":
            length = 1000 + 3000 * random_nums[2]
            depth = 2000 + 2000 * random_nums[3]
            width = 2000 + 3000 * random_nums[2]

            kwargs = {
                "strike": 180 * random_nums[6],
                "dip": [45, 90][random_nums[7] < 0.5],
                "length": length,
                "rake": [-90, -90][random_nums[7] < 0.5],
                "slip": 3,
                "top_depth": top_depth,
                "bottom_depth": top_depth + (top_depth * 2 + 10000 * random_nums[8]),
                "width": depth / 4,
                "depth": depth,
                "opening": 5,
            }

        elif event_type == "sill":
            length = 2000 + 3000 * random_nums[2]
            depth = 4000 + 2000 * random_nums[3]
            width = 2000 + 3000 * random_nums[4]

            kwargs = {
                "strike": 180 * random_nums[5],
                "dip": 0,
                "length": length,
                "rake": 0,
                "slip": 0,
                "top_depth": 0,
                "bottom_depth": 0,
                "width": width,
                "depth": depth,
                "opening": 0.5,
            }

        elif event_type == "dyke":
            length = 2000 + 3000 * random_nums[2]
            depth = 4000 + 2000 * random_nums[3]
            width = 2000 + 3000 * random_nums[4]

            kwargs = {
                "strike": 180 * random_nums[5],
                "dip": 0,
                "length": length,
                "rake": 0,
                "slip": 0,
                "top_depth": top_depth,
                "bottom_depth": top_depth + (top_depth * 2 + 10000 * random_nums[8]),
                "width": width,
                "depth": depth,
                "opening": 0.5,
            }

    else:
        axes_max = (tile_size) * 90

        top_depth = kwargs["top_depth"]
        source_x = kwargs["source_x"]
        source_y = kwargs["source_y"]
        length = kwargs["length"]

    start = time.perf_counter()
    Event = Okada(event_type, (source_x, source_y), tile_size=tile_size, **kwargs)
    end = time.perf_counter()

    los_grid = (
        Event.los_displacement * amplitude_scalar * [-1, -1][random_nums[7] < 0.5] * 0.5
    )

    masked_indices = np.abs(los_grid) >= np.pi * 2
    n_masked_indices = np.abs(los_grid) < np.pi * 2
    no_masked_indices = los_grid < np.pi * 2

    if event_type == "quake":
        los_grid[no_masked_indices] = 0
    masked_grid[masked_indices] = 1

    atmosphere_phase = aps_simulate(tile_size) * atmosphere_scalar

    coherence_mask = coherence_mask_simulate(tile_size, threshold=random_nums[8] * 0.3)
    coh_masked_indices = coherence_mask[0, 0:tile_size, 0:tile_size] == 0

    interferogram = los_grid + atmosphere_phase[0:tile_size, 0:tile_size]

    n_masked_indices = np.abs(interferogram) < np.pi * 5
    n_masked_indices2 = np.abs(interferogram) < np.pi * 7
    masked_grid[masked_indices] = 1
    masked_grid[n_masked_indices2] = 1
    masked_grid[n_masked_indices] = 0
    # masked_grid[coh_masked_indices]    = 0
    interferogram[coh_masked_indices] = 0

    wrapped_grid = np.angle(np.exp(1j * (interferogram)))

    # zeros = wrapped_grid == 0
    # wrapped_grid += np.pi
    # wrapped_grid /= (2 * np.pi)
    # wrapped_grid[zeros] = 0

    if log:
        print("__________\n")
        print(event_type)
        print("__________\n")
        print("Length         (meters)  ", length)
        print("Top Depth      (meters)  ", top_depth)
        print("Bottom Depth   (meters)  ", kwargs["bottom_depth"])
        print("Depth          (meters)  ", kwargs["depth"])
        print("Width          (meters)  ", kwargs["width"])
        print("")
        print("Slip           (0  or 1) ", kwargs["slip"])
        print("Dip            (degrees) ", kwargs["dip"])
        print("Rake           (degrees) ", kwargs["rake"])
        print("Strike         (degrees) ", kwargs["strike"])
        print("Opening:       (meters)  ", kwargs["opening"])
        print("")
        print("Max X Position (meters)  ", axes_max)
        print("Max Y Position (meters)  ", axes_max)
        print("Src X Position (meters)  ", source_x)
        print("Src Y Position (meters)  ", source_y)
        print("")
        print("Maximum Phase Value:     ", np.max(np.abs(interferogram)))
        print("")
        print("Compute Time   (seconds) ", end - start)
        print("__________\n")

    unwrapped = (
        Event.los_displacement * amplitude_scalar
        + atmosphere_phase[0:tile_size, 0:tile_size]
    )

    return unwrapped, masked_grid, wrapped_grid, presence


def gen_gaussian_noise(
    seed: int = 0,
    tile_size: int = 512,
    noise_level: float = 90 * np.pi,
    threshold: float = 90 * np.pi / 4,
):
    if seed != 0:
        np.random.seed(seed)

    noise_grids = np.random.uniform(
        -noise_level, noise_level, size=(2, tile_size, tile_size)
    )

    inconsistancy = np.abs(noise_grids[1]) >= threshold

    noise_grids[1][inconsistancy] = 0

    index = noise_grids[1] == 0
    noise_grids[0][index] = 0

    return noise_grids[0]


def gen_sim_noise(
    seed: int = 0,
    tile_size: int = 512,
    gaussian_only: bool = False,
    atmosphere_scalar: float = 90 * np.pi,
):
    """
    Generate a wrapped interferogram along with an event-mask simulating a noisy interferogram with no deformation.

    Parameters
    -----------
    seed : int, Optional
        A seed for the random functions. For the same seed, with all other values the same
        as well, the interferogram generation will have the same results. If left at 0,
        the results will be different every time.
    tile_size : int, Optional
        The desired dimensional size of the interferogram pairs. This should match
        the input shape of the model.
    log : bool, Optional
        If true, the function will log various relevant values in the console.
    atmosphere_scalar : float, Optional
        Scale factor for the intensity of atmospheric noise.

    Returns
    --------
    masked_grid : np.ndarray(shape=(tile_size, tile_size))
        An array representing a mask over the simulated deformation which simulates masking an event.
    wrapped_grid : np.ndarray(shape=(tile_size, tile_size)
        The wrapped interferogram.
    presence : [1] or [0]
        [1] if the image contains an event else [0]
    """

    if seed != 0:
        np.random.seed(seed)

    presence = np.asarray([0])

    if gaussian_only:
        threshold = 90 * np.pi
        noise_grid_full = gen_gaussian_noise(seed, tile_size, threshold=threshold)

        noise_grid_inconsistent = gen_gaussian_noise(
            seed, tile_size, threshold=(threshold * np.random.random() / 2)
        )

        threshold = np.random.random() / 2
        coherence_mask = coherence_mask_simulate(tile_size, threshold=threshold)
        coh_indices = coherence_mask[0, 0:tile_size, 0:tile_size] == 0

        noise_grid = noise_grid_full
        noise_grid[coh_indices] = noise_grid_inconsistent[coh_indices]

        wrapped_grid = np.angle(np.exp(1j * (noise_grid)))

        masked_grid = np.zeros((tile_size, tile_size))
        phase = masked_grid

    else:
        simulated_topography = gen_fake_topo(
            size=tile_size, alt_scale_min=50, alt_scale_max=100
        )

        turb_phase = aps_simulate(tile_size) * atmosphere_scalar
        topo_phase = (
            atm_topo_simulate(simulated_topography) * atmosphere_scalar
        )  # aps_simulate(tile_size) * atmosphere_scalar +

        noise_grid_full = gen_gaussian_noise(seed, tile_size, threshold=90 * np.pi)
        coherence_mask = coherence_mask_simulate(tile_size, threshold=0.3)
        coh_indices = coherence_mask[0, 0:tile_size, 0:tile_size] == 0

        phase = turb_phase + topo_phase
        phase[coh_indices] = noise_grid_full[coh_indices]

        wrapped_grid = np.angle(np.exp(1j * (phase)))

        masked_grid = np.zeros((tile_size, tile_size))

    # zeros = wrapped_grid == 0
    # wrapped_grid += np.pi
    # wrapped_grid /= (2 * np.pi)
    # wrapped_grid[zeros] = 0

    return phase, masked_grid, wrapped_grid, presence


def gen_simulated_time_series(
    n_interferograms: int = 32,
    tile_size: int = 512,
    seed: int = 0,
    atmosphere_scalar: float = 90 * np.pi,
    noise_only: bool = False,
):
    """
    Generate a time-series of interferograms with simulated deformation. Correlated by a common dem.

    Parameters
    -----------
    seed : int, Optional
        A seed for the random functions. For the same seed, with all other values the same
        as well, the interferogram generation will have the same results. If left at 0,
        the results will be different every time.
    tile_size : int, Optional
        The desired dimensional size of the interferogram pairs. This should match
        the input shape of the model.
    log : bool, Optional
        If true, the function will log various relevant values in the console.
    atmosphere_scalar : float, Optional
        Scale factor for the intensity of atmospheric noise.

    Returns
    --------
    masked_grid : np.ndarray(shape=(tile_size, tile_size))
        An array representing a mask over the simulated deformation which simulates masking an event.
    wrapped_grid : np.ndarray(shape=(tile_size, tile_size)
        The wrapped interferogram.
    presence : [1] or [0]
        [1] if the image contains an event else [0]
    """

    if seed != 0:
        np.random.seed(seed)

    simulated_topography = gen_fake_topo(size=tile_size, alt_scale_max=100)

    event_type = "quake"
    axes_max = (tile_size) * 90
    random_nums = np.random.rand(13)
    scalar = 100 * np.pi
    mask = np.zeros((tile_size, tile_size))

    if not noise_only:
        axes_max = (tile_size) * 90

        source_x = axes_max // ((random_nums[0] * 10) + 1)
        source_y = axes_max // ((random_nums[1] * 10) + 1)

        length = 1000 + 3000 * random_nums[2]
        top_depth = 6000 + 5000 * random_nums[3]
        depth = 2000 + 2000 * random_nums[3]

        kwargs = {
            "strike": 180 * random_nums[6],
            "dip": [45, 90][int(random_nums[7] < 0.5)],
            "length": length,
            "rake": [-90, -90][int(random_nums[7] < 0.5)],
            "slip": 3,
            "top_depth": top_depth,
            "bottom_depth": top_depth + (top_depth * 2 + 10000 * random_nums[8]),
            "width": depth / 4,
            "depth": depth,
            "opening": 5,
        }

        if event_type == "dyke":
            kwargs["dip"] = 90

        Event = Okada(event_type, (source_x, source_y), tile_size=tile_size, **kwargs)

        los_displacement = Event.los_displacement

        phase = scalar * los_displacement

        mask[np.abs(phase) > np.pi] = 1

    else:
        los_displacement = np.zeros((tile_size, tile_size))

    phases = np.zeros((n_interferograms, 2, tile_size, tile_size))

    inflection = (
        random.random() * 0.6 + 0.2
    )  # set the inflection point to a point between 0.2 and 0.8
    inflection_rate = (
        random.random() * 2 + 2
    )  # set the inflection rate to be between 2 and 4
    inflection_point = n_interferograms * inflection
    step_size = (
        inflection_point + (n_interferograms - inflection_point) * inflection_rate
    )
    displacement = los_displacement / step_size

    for i in range(n_interferograms):
        topo_phase = np.abs(
            atm_topo_simulate(simulated_topography) * atmosphere_scalar * 0.15 * np.pi
        )
        turb_phase = aps_simulate(tile_size) * atmosphere_scalar * 0.3
        displacement_step = displacement * i
        if i > inflection_point:
            displacement_step = displacement_step + displacement * inflection_rate * (
                i - inflection_point
            )

        phase_step = scalar * displacement_step * i + topo_phase + turb_phase
        wrapped_phase_step = np.angle(np.exp(1j * (phase_step)))

        phase_step = (phase_step + np.abs(np.min(phase_step))) / np.max(
            (phase_step + np.abs(np.min(phase_step)))
        )  # Normalize

        phases[i] = [phase_step, wrapped_phase_step]

    return phases, mask
