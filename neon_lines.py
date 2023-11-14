"""
Data analysis ...
"""

from astropy.io import fits
from lmfit.models import Model
from scipy import signal
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np

def neondata():
    """ Takes 10 different measurements of the neon absorption spectrum.
    Returns the average of these measurements as a 1D-array
    """
    
    neon_set = []
    for i in range(1,10):
        
        data = fits.getdata(f'/home/gideon/Documents/NSP2/LISA data/Verschillende hoogtes/Sky_angles/calibration/neon-00{i}.fit')
        neon_set.append(data)
        
    neon_set.append(fits.getdata('/home/gideon/Documents/NSP2/LISA data/Verschillende hoogtes/Sky_angles/calibration/neon-010.fit'))
    
    reduced_neon_set = []
    for j in neon_set:
        reduced_data = np.sum(j, axis=0)
        reduced_neon_set.append(reduced_data)
        
    reduced_neon_set = np.array(reduced_neon_set)
    avg = np.mean(reduced_neon_set, axis=0)

    return avg
    
# crop to effective data, plot
data = neondata()[735:1320]
plt.figure()
plt.plot(data, 'o', markersize = 0.75)
# known neon absorption peaks in angstrom, converted to nm
known_neon_lines = 0.1 * np.array([5852.49, 5881.89, 5944.83, 5975.53, 6030.00, 6074.34, 6096.16, 6143.06, 6163.59, 6217.28, 6266.49, 6304.79, 6334.43, 6382.99, 6402.25, 6506.53, 6532.88, 6598.95, 6678.28, 6717.04, 6929.47, 7032.41, 7173.94, 7245.17, 7438.90])

# convert to nm
known_neon_lines = np.delete(known_neon_lines, np.s_[-2:])

pixel_peaks = signal.find_peaks(data, height=400000)[0]



def function(x,a,b,c):
    
     return a*x**2 + b*x + c

model = Model(function)
result = model.fit(known_neon_lines, x=pixel_peaks, a=-300000,b=1,c=600)

plt.figure()
plt.plot(pixel_peaks, known_neon_lines, 'o')
plt.plot(pixel_peaks, result.best_fit)
plt.xlabel('pixels')
plt.ylabel('wavelength (nm)')

a = result.params['a'].value
b = result.params['b'].value
c = result.params['c'].value
