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
from scipy.signal import find_peaks
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
    
    def __init__(self, data_folder,degrees,measurement, min,max,wavelength, width, smoothing):
        
        meting = Normalize(data_folder, degrees, measurement)
        meting.isolate(min,max)
        meting.mask_peak(wavelength, width)
        meting.smooth_function(smoothing)
        meting.curve_fit()
        self.min = min
        self.max = max
        self.x, self.y = meting.normalize()
        

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

    # drop points above 1 
    def drop(self):
        
        y_diff = 1 - self.y
        
        bool = np.where(y_diff>0)
        self.y = self.y[bool]
        self.x = self.x[bool]
        return self.x, self.y
    
    def trap(self):
        
        # transforms function for proper integral
        self.y = -1*self.y + 1
        
        plt.figure()
        plt.plot(self.x, self.y, 'o')
        
        area = trapezoid(x=self.x, y=self.y)
        return area
    
degrees = str('30')
measurement = str('006')

area = []
mean_list = []
error_list = []

for i in range(1,9):
    
    measurement = str(f'00{i}')
    meting = Area(Path(r'C:\Users\Femke\Desktop\NS_jaar_3\Periode_2\NSP2\data_mapje\LISA data\Verschillende hoogtes\Sky_angles\Sky_angles'), degrees, measurement, 640, 670, 656, 0.5, 10)
    meting.isolate(655,660)
    meting.drop()
    area.append(meting.trap())
    
mean = np.mean(np.array(area))
print(mean)
err = np.std(np.array(area)) / np.sqrt(8)
print(err)

data = {
  "degrees":[] ,
  "mean area": mean,
  "error": err,

}

#load data into a DataFrame object:
df = pd.DataFrame(data)

print(df) 