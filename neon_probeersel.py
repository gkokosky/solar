# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 11:27:35 2023

@author: Femke
"""

"""
Data analysis ...
"""

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np


neon_set = []
for i in range(1,10):
    
    data = fits.getdata(f'C:\Users\Femke\Desktop\NS_jaar_3\Periode_2\NSP2\data_mapje\LISA data\Verschillende hoogtes\Sky_angles\calibration\neon-001')
    neon_set.append(data)
    
plt.imshow(neon_set[1], cmap = 'gray', norm='log')

print(neon_set[0])