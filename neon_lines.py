"""
Data analysis ...
"""

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np


neon_set = []
for i in range(1,10):
    
    data = fits.getdata(f'/home/gideon/Documents/NSP2/LISA data/Verschillende hoogtes/Sky_angles/calibration/neon-00{i}.fit')
    neon_set.append(data)
    
#plt.imshow(neon_set[1], cmap = 'gray', norm='log')

data = neon_set[0]
print(data)
data = data[:,700:1400]

intensity_profile = np.sum(data, axis=0)
print(intensity_profile)
plt.plot(intensity_profile)

neon_lines = []