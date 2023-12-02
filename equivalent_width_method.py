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
from scipy.integrate import simpson
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
    
    def __init__(self, degrees,measurement, min,max,wavelength, width, smoothing, small_min, small_max):
        
        meting = Normalize(degrees, measurement)
        meting.isolate(min,max)
        meting.mask_peak(wavelength, width)
        meting.smooth_function(smoothing)
        meting.curve_fit()
        
        self.measurement = measurement
        
        # converts measurement from int to string for data reading
        for i in range(1,11):
            degrees = f'{degrees}'
            measurement = f'{i}'
            if len(measurement) == 1:
                measurement = f'00{i}'
                self.measurement = f'00{i}'
            elif len(measurement) == 2:
                measurement = f'0{i}'
            else:
                print('huh')
                
        # converts angle from int to str for same purpose
        if degrees == 6:
            degrees = f'0{degrees}'
        else:
            degrees = f'{degrees}'
        
        
        self.width = width
        self.min = small_min
        self.max = small_max
        self.x, self.y = meting.normalize()
        
        self.wavelength = wavelength

    # def peak(self):
        
    #     x = self.x
    #     y = self.y
    #     wavelength = self.wavelength
        
    #     peaks, _ = find_peaks(-y)
    #     peak_diff = np.abs(x[peaks] - wavelength)
    #     peak = np.argmin(peak_diff)
    #     peak = peaks[peak]
    #     width, _, _, _ = peak_widths(-y, np.array([peak]))
        
    #     width = self.width * width
    #     # find leftmost part of peak
    #     x_left = x[peak] - width
    #     left_diff = np.abs(x - x_left)
    #     left_idx = np.argmin(left_diff)
        
    #     # find rightmost part of peak
    #     x_right = x[peak] + width
    #     right_diff = np.abs(x - x_right)
    #     right_idx = np.argmin(right_diff)
        
    #     self.x = x[left_idx: right_idx+1]
    #     self.y = y[left_idx: right_idx+1]
        
    #     return self.x, self.y
    
    #isolates peak through manual wavelength input 
    def peak(self):
        
        x = self.x
        y = self.y
        min = self.min
        max = self.max
        
        # finds smallest difference between x and min/max
        min_diff = np.abs(self.x - min)
        max_diff = np.abs(self.x - max)
        
        # finds index associated with these values
        min_idx = min_diff.argmin()
        max_idx = max_diff.argmin()
        
        self.x = self.x[min_idx:max_idx+1]
        self.y = self.y[min_idx:max_idx+1]
        
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