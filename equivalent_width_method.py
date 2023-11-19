# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 14:06:46 2023

@author: Femke
"""
# equivalent width method script
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline
import matplotlib.pyplot as plt
plt.style.use('seaborn-talk')

# MatPlotLib tools for drawing on plots.
import matplotlib.transforms as mtransforms
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
# AstroPy allows Python to perform common astronomial tasks.
from astropy.io import fits
from astropy import units as u
from astropy.visualization import quantity_support
quantity_support()
# comes in handy when determinign the eq. width
from astropy.utils.data import download_file
from specutils import Spectrum1D
from specutils import SpectralRegion
from specutils.analysis import equivalent_width
from specutils.analysis import fwhm
from astropy.io import fits
from lmfit.models import Model
from scipy.signal import find_peaks
from pathlib import Path

# DATA IMPORTEREN
def neondata():
    """ Takes 10 different measurements of the neon absorption spectrum.
    Returns the average of these measurements as a 1D-array
    """
    
    neon_set = []
    for i in range(1,10):
        
        p = Path(__file__).with_name(f'neon-00{i}.fit')
        name = p.absolute()
        data = fits.getdata(name)
        neon_set.append(data)
        
    p_final = Path(__file__).with_name('neon-010.fit')
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


peaks, _ = find_peaks(y, height=500000)
plt.figure()
plt.plot(x,y)
plt.plot(x[peaks],y[peaks],'o')

neon_lines = [5852.49, 5891.89, 5944.83, 5975.53, 6030.00, 6074.34, 6096.16, 6143.06, 6163.59, 6217.28, 6266.49, 6304.79, 6334.43, 6382.99,6402.25,6506.53, 6532.88,6598.95]

# FIT EN PLOT
def function(x,a,b,c):
    
     return a*x**2 + b*x + c

pixel_peaks = x[peaks]
model = Model(function)
result = model.fit(neon_lines, x=pixel_peaks, a=1,b=1,c=1)

plt.figure()
plt.plot(pixel_peaks, neon_lines, 'o')
plt.plot(pixel_peaks, result.best_fit)
plt.xlabel('pixels')
plt.ylabel('wavelength (nm)')
plt.savefig('pixel_to_wavelength.png')

# plot of residuals (which we do not have yet)
# plt.figure()
# plt.plot(pixel_peaks,result.residual,'o')

a = result.params['a'].value
b = result.params['b'].value
c = result.params['c'].value


# Rechthoek maken
# Plot one of our stars and annotate the equivalent width.
fig2, ax2 = plt.subplots()
fig2.suptitle(' Equivalent Width', fontsize='24')

ax2.plot(wave1, flux1)

plt.xlabel('λ (Å)',fontsize='20')
plt.ylabel('Relative flux', fontsize='20')

ax2.set_xlim([6643.3,6644])

# Fill in the area under the curve and overlay a rectangle of equal area.
# Green area
trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
ax2.fill_between(wave1, 1, flux1, where=flux1 < 1*u.Unit('erg cm-2 s-1 AA-1'),
                 facecolor='green', interpolate=True, alpha=0.3)
# Dark gray area.
ax2.add_patch(plt.Rectangle((.29, .045), .358, 0.85, ec='k', fc="k",
                            alpha=0.3,  transform=ax2.transAxes))

plt.axhline(y=1,ls='--',color='red',lw=2)

plt.show()

# EQUIVALENT WIDTH BEPALEN 
# hierbij word gebruik gemaakt van de specutils package

# Calculate the EW of Ni I and Eu II in their specific spectral regions.
ni1 = equivalent_width(spec1, regions=SpectralRegion(6643.0*u.AA,6644*u.AA))
eu1 = equivalent_width(spec1, regions=SpectralRegion(6645.0*u.AA,6645.5*u.AA))
r1 = eu1/ni1

# Rounding
ni1 = np.round(ni1,3)
eu1 = np.round(eu1,3)
r1 = np.round(r1,3)

# Print the Equivalent width and ratio
print('EW Ni I 6643A\tEW Eu II 6645 A\tEu/Ni\tName') # Print a header row
print(str(ni1)+'\t'+str(eu1)+'\t'+str(r1)+'\t'+label1) # Print for target 1
