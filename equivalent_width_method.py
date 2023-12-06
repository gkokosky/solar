# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 14:06:46 2023

@author: Femke
"""
from normalize import Normalize

# equivalent width method script
import numpy as np
import matplotlib.pyplot as plt


class Area:
    def __init__(
        self,
        degrees,
        measurement,
        min,
        max,
        wavelength,
        width,
        smoothing,
        small_min,
        small_max,
    ):
        meting = Normalize(degrees, measurement)
        meting.isolate(min, max)
        meting.mask_peak(wavelength, width)
        meting.smooth_function(smoothing)
        meting.curve_fit()

        # converts measurement from int to string for data reading
        for i in range(1, 11):
            degrees = f"{degrees}"
            measurement = f"{i}"
            if len(measurement) == 1:
                measurement = f"00{i}"
                self.measurement = f"00{i}"
            elif len(measurement) == 2:
                measurement = f"0{i}"
            else:
                print("huh")

        self.measurement = measurement

        # converts angle from int to str for same purpose
        if degrees == 6:
            degrees = f"0{degrees}"
        else:
            degrees = f"{degrees}"

        self.width = width
        self.small_min = small_min
        self.small_max = small_max
        self.x, self.y = meting.normalize()

        self.wavelength = wavelength
        self.wavelength = wavelength

    # isolates peak through manual wavelength input
    def peak(self):
        x = self.x
        y = self.y
        min = self.small_min
        max = self.small_max

        # finds smallest difference between x and min/max
        min_diff = np.abs(self.x - min)
        max_diff = np.abs(self.x - max)

        # finds index associated with these values
        min_idx = min_diff.argmin()
        max_idx = max_diff.argmin()

        self.x = self.x[min_idx : max_idx + 1]
        self.y = self.y[min_idx : max_idx + 1]

        return self.x, self.y

    # drop points above 1
    # def drop(self):

    #     y_diff = 1 - self.y

    #     bool = np.where(y_diff>0)
    #     self.y = self.y[bool]
    #     self.x = self.x[bool]
    #     return self.x, self.y

    def trap(self):
        # transforms function for proper integral
        x = self.x
        y = -self.y + 1
        area = np.trapz(y=y, x=x)

        return area
