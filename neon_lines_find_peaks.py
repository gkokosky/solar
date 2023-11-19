"""
Data analysis ...
"""

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np


neon_set = []
for i in range(1,10):
    
    data = fits.getdata(f'C:\Users\Femke\Desktop\NS_jaar_3\Periode_2\NSP2\data_mapje\LISA data\Verschillende hoogtes\Sky_angles')
    neon_set.append(data)
    
#plt.imshow(neon_set[1], cmap = 'gray', norm='log')

data = neon_set[0]
data = data[:,700:1400]

intensity_profile = np.sum(data, axis=0)
plt.plot(intensity_profile)
plt.xlim(0,200)


print(intensity_profile)
