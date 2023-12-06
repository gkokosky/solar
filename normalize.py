from neon_lines import a, b, c
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import gaussian_filter1d
from lmfit import Model


class Normalize:
    def __init__(self, degrees, measurement):
        """Initializes data from 1 measurement. Reduces to 1D-data and
        converts pixels to wavelength in nm

        Args:
            degrees (int): angle at which the telescope was pointed
            measurement {int}: number of measurement (01-10)
            data_folder (str): folder in which the 'fits' files are stored
        """
        self.x = np.array([])
        self.measurement = measurement
        self.y = np.array([])
        self.x_masked = np.array([])
        self.y_masked = np.array([])
        self.smooth_y = np.array([])

        for i in range(1, 11):
            measurement = f"{i}"
            if len(measurement) == 1:
                measurement = f"00{i}"
                self.measurement = f"00{i}"
            elif len(measurement) == 2:
                measurement = f"0{i}"
            else:
                print("huh")

        # converts angle from int to str for same purpose
        if degrees == 6:
            degrees = f"0{degrees}"
        else:
            degrees = f"{degrees}"

        file = Path("Sky_angles/Sky_angles", f"{degrees}deg-{measurement}.fit")
        self.data = fits.getdata(file)

        self.a = 0
        self.b = 0
        self.c = 0

        reduced_data = []
        reduced_data = np.sum(self.data, axis=0)

        # get dark measurement and reduce
        dark_file = Path("Sky_angles/dark", "dark-007_half_s_.fit")
        self.dark = fits.getdata(dark_file)
        reduced_dark = []
        reduced_dark = np.sum(self.dark, axis=0)

        y_pixel = np.array(reduced_data) - reduced_dark
        x_pixel = np.array([i for i in range(len(y_pixel))])

        self.x = a * x_pixel**2 + b * x_pixel + c
        self.x = self.x * 0.1
        self.y = np.array(y_pixel)

        if degrees == "06" and measurement == "001":
            self.graph_bool = True
        else:
            self.graph_bool = False

    def isolate(self, min, max):
        """Isolates specific wavelength range, so that only one absorption peak is found.

        Args:
            min (float): lowest wavelength in range to analyze
            max (float): highest wavelength in range to analyze
        """

        # finds smallest difference between x and min/max
        min_diff = np.abs(self.x - min)
        max_diff = np.abs(self.x - max)

        # finds index associated with these values
        min_idx = min_diff.argmin()
        max_idx = max_diff.argmin()

        self.x = self.x[min_idx : max_idx + 1]
        self.y = self.y[min_idx : max_idx + 1]

        return self.x, self.y

    def mask_peak(self, wavelength, width_multiplier):
        """Finds the absorption peak, and masks it for proper
        normalization.
        """

        x = self.x
        y = self.y

        peaks, _ = find_peaks(-y)

        # finds peak associated with given wavelength
        peak_diff = np.abs(x[peaks] - wavelength)
        peak = np.argmin(peak_diff)
        peak = peaks[peak]
        width, _, _, _ = peak_widths(-y, np.array([peak]))

        width = width_multiplier * width

        # find leftmost part of peak
        x_left = x[peak] - width
        left_diff = np.abs(x - x_left)
        left_idx = np.argmin(left_diff)

        # find rightmost part of peak
        x_right = x[peak] + width
        right_diff = np.abs(x - x_right)
        right_idx = np.argmin(right_diff)

        mask_range = np.array(range(left_idx, right_idx + 1))
        mask = np.ones_like(x, dtype=bool)

        mask[mask_range] = False

        self.x_masked = x[mask]
        self.y_masked = y[mask]

        return self.x_masked, self.y_masked

    def smooth_function(self, smoothing):
        y = self.y_masked
        self.smooth_y = gaussian_filter1d(y, sigma=smoothing)
        return self.smooth_y

    def curve_fit(self):
        x = self.x_masked
        y = self.smooth_y

        def function(x, a, b, c):
            return a * x**2 + b * x + c

        model = Model(function)
        pars = model.make_params(a=1, b=1, c=1)

        result = model.fit(y, pars, x=x)

        self.a = result.params["a"].value
        self.b = result.params["b"].value
        self.c = result.params["c"].value

        return self.a, self.b, self.c

    def normalize(self):
        x = self.x
        y = self.y

        y_fit = np.array(self.a * x**2 + self.b * x + self.c)

        y_norm = y / y_fit

        return self.x, y_norm


meting = Normalize(10, 1)
x, y = meting.isolate(525, 535)
plt.figure()
plt.plot(x, y)
