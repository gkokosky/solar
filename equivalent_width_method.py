# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 14:06:46 2023

@author: Femke
"""
from normalize import Normalize
# equivalent width method script
import numpy as np
import matplotlib.pyplot as plt
# AstroPy allows Python to perform common astronomial tasks.
from astropy.visualization import quantity_support
quantity_support()
# comes in handy when determinign the eq. width
from lmfit.models import GaussianModel
from scipy.signal import find_peaks, peak_widths
from scipy.integrate import trapezoid
from pathlib import Path
import pandas as pd
        
    
# data_folder = Path('/home/gideon/Documents/NSP2/LISA data/Verschillende hoogtes/Sky_angles/Sky_angles')
# degrees = str('30')
# measurement=str('010')

# meting = Normalize(data_folder, degrees, measurement)
# meting.isolate(640,670)
# meting.mask_peak(656, 0.5)
# meting.smooth_function(10)
# meting.curve_fit()
# x, y = meting.normalize()

# plt.figure()
# plt.plot(x,y,'o')
# plt.xlabel('golflengte (nm)')
# plt.ylabel('relatieve intensiteit')
# plt.savefig('normalized.png', dpi=300)

class Area:
    
    def __init__(self, degrees,measurement, min,max,wavelength, width, smoothing):
        
        meting = Normalize(degrees, measurement)
        meting.isolate(min,max)
        meting.mask_peak(wavelength, width)
        meting.smooth_function(smoothing)
        meting.curve_fit()
        self.min = min
        self.max = max
        self.x, self.y = meting.normalize()
        
        self.wavelength = wavelength
        

    # isolate further for peak only
    def isolate(self, min,max):
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
        
        self.x = self.x[min_idx:max_idx+1]
        self.y = self.y[min_idx:max_idx+1]
        
        return self.x, self.y

    def peak(self):
        
        x = self.x
        y = self.y
        wavelength = self.wavelength
        
        peaks, _ = find_peaks(-y)
        peak_diff = np.abs(x[peaks] - wavelength)
        peak = np.argmin(peak_diff)
        peak = peaks[peak]
        width, _, _, _ = peak_widths(-y, np.array([peak]))
        
        width = 0.5 * width
        # find leftmost part of peak
        x_left = x[peak] - width
        left_diff = np.abs(x - x_left)
        left_idx = np.argmin(left_diff)
        
        # find rightmost part of peak
        x_right = x[peak] + width
        right_diff = np.abs(x - x_right)
        right_idx = np.argmin(right_diff)
        
        self.x = x[left_idx: right_idx+1]
        self.y = y[left_idx: right_idx+1]
        
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
        y = -1*self.y + 1
        
        area = trapezoid(y=y)
        return area