"""
Data analysis ...
"""

from astropy.io import fits 
from lmfit.models import Model
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
#from statsmodels.formula.api.models import ols

def neondata():
    """ Takes 10 different measurements of the neon absorption spectrum.
    Returns the average of these measurements as a 1D-array
    """
    
    neon_set = []
    for i in range(1,10):
        
        p = Path(r'Sky_angles/calibration', f'neon-00{i}.fit')
        name = p.absolute()
        data = fits.getdata(name)
        neon_set.append(data)
        
    p_final = Path(r'Sky_angles/calibration', 'neon-010.fit')
    name_final = p_final.absolute()
    neon_set.append(fits.getdata(name_final))
    
    reduced_neon_set = []
    for j in neon_set:
        reduced_data = np.sum(j, axis=0)
        reduced_neon_set.append(reduced_data)
        
    reduced_neon_set = np.array(reduced_neon_set)
    avg = np.mean(reduced_neon_set, axis=0)
    err = np.std(reduced_neon_set, axis=0) / np.sqrt(10)

    return avg, err
    
# crop to effective data, plot
data, err = neondata()
# make data x and y instead of just y
x = np.array([i for i in range(len(data))])[735:1050]
y = np.array(data)[735:1050]
y_err = np.array(err)[735:1050]


peaks, _ = find_peaks(y, height=500000)
peaks = np.delete(peaks, 0)
plt.figure()
plt.title('pieken neon kalibratie spectrum')
plt.plot(x,y)
plt.plot(x[peaks],y[peaks],'o')
plt.show()
neon_lines = [5852.49, 5891.89, 5944.83, 5975.53, 6030.00, 6074.34, 6096.16, 6143.06, 6163.59, 6217.28, 6266.49, 6304.79, 6334.43, 6382.99,6402.25,6506.53, 6532.88,6598.95]
neon_lines = np.delete(np.array(neon_lines), 0)

def function(x,a,b,c):
    
     return a*x**2 + b*x + c

pixel_peaks = x[peaks]
x_err = y_err[peaks]
model = Model(function)
result = model.fit(neon_lines, x=pixel_peaks, a=1,b=1,c=1)

plt.figure()
plt.title('golflengte vs pixels')
plt.plot(pixel_peaks, neon_lines, 'o')
plt.plot(pixel_peaks, result.best_fit)
plt.xlabel('pixels')
plt.ylabel('golflengte (nm)')
plt.show()

plt.figure()
plt.plot(pixel_peaks,result.residual,'o')

a = result.params['a'].value
b = result.params['b'].value
c = result.params['c'].value


# def calculate_residuals():
#     residual_list = []
#     # residual is verschil tussen fit-waarde en neon waarde
#     for i in known_neon_lines_unmultiplied:
#         error = np.abs(i - result.best_fit)
#         residual_list.append(error)
            
#     return residual_list

# # residual plot
# plt.figure()
# plt.title("residual plot")
# plt.plot(pixel_peaks, calculate_residuals(), 'o')
# plt.xlabel('x-value = pixel')
# plt.ylabel('y-value = residual')
# plt.show()

