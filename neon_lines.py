"""
Data analysis ...
"""

from astropy.io import fits
import matplotlib.pyplot as plt


neon_set = []
for i in range(1,2):
    
    data = fits.getdata(f'/home/gideon/Documents/NSP2/LISA data/Verschillende hoogtes/Sky_angles/calibration/neon-00{i}.fit')
    neon_set.append(data)
    
print(neon_set)