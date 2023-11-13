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

known_neon_lines = [5852.49, 5881.89, 5944.83, 5975.53, 6030.00, 6074.34, 6096.16, 6143.06, 6163.59, 6217.28, 6266.49, 6304.79, 6334.43, 6382.99, 6402.25, 6506.53, 6532.88, 6598.95, 6678.28, 6717.04, 6929.47, 7032.41, 7173.94, 7245.17, 7438.90]