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

# NORMALISATIE
from neon_lines import a, b, c
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import gaussian_filter1d
from lmfit import Model

class Normalize:
    
    def __init__(self,data_folder,degrees,measurement):
        """ Initializes data from 1 measurement. Reduces to 1D-data and
        converts pixels to wavelength in nm
        
        Args:
            degrees (int): angle at which the telescope was pointed
            measurement {int}: number of measurement (01-10)
            data_folder (str): folder in which the 'fits' files are stored
        """
        self.x = np.array([])
        self.y = np.array([])
        self.x_masked = np.array([])
        self.y_masked = np.array([])
        self.smooth_y = np.array([])
        
        self.data = fits.getdata(f'{data_folder}/{degrees}deg-{measurement}.fit')
        
        self.a = 0
        self.b = 0
        self.c = 0
        
        reduced_data = []
        reduced_data = np.sum(self.data, axis=0)
            
        y_pixel = np.array(reduced_data)
        x_pixel = np.array([i for i in range(len(y_pixel))])

        self.x = a * x_pixel**2 + b * x_pixel + c
        self.x = self.x * 0.1
        self.y = np.array(y_pixel)


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
        
    def mask_peak(self, wavelength):
        """Finds the absorption peak, and masks it for proper
        normalization.
        """
        
        x = self.x
        y = self.y
        
        peaks, _ = find_peaks(-y)
        
        peak_diff = np.abs(x[peaks] - wavelength)
        peak = np.argmin(peak_diff)
        peak = peaks[peak]
        width, _, _, _ = peak_widths(-y, np.array([peak]))
        
        width = 0.5*width
        
        # find leftmost part of peak
        x_left = x[peak] - width
        left_diff = np.abs(x - x_left)
        left_idx = np.argmin(left_diff)
        
        # find rightmost part of peak
        x_right = x[peak] + width
        right_diff = np.abs(x - x_right)
        right_idx = np.argmin(right_diff)
        
        mask_range = np.array(range(left_idx, right_idx+1))
        mask = np.ones_like(x, dtype=bool)

        mask[mask_range] = False
        
        self.x_masked = x[mask]
        self.y_masked = y[mask]
        
        return self.x_masked, self.y_masked
        

    def smooth_function(self):
        
        y = self.y_masked
        self.smooth_y = gaussian_filter1d(y,sigma=3)
        return self.smooth_y        

    def curve_fit(self):
        
        x = self.x_masked
        y = self.smooth_y
        
        def function(x, a, b, c):
            return a * x**2 + b*x + c
        
        model = Model(function)
        pars = model.make_params(a=1,b=1,c=1)
        
        result = model.fit(y, pars,x=x)
        
        self.a = result.params['a'].value
        self.b = result.params['b'].value
        self.c = result.params['c'].value
        
        return self.a,self.b,self.c
    
    def normalize(self):
        
        x = self.x
        y = self.y
        
        y_fit = np.array(self.a * x **2 + self.b * x + self.c)

        plt.figure()
        plt.plot(x,y,'o')
        plt.plot(x,y_fit)
        
        y_norm = y / y_fit
        return y_norm
        
data_folder = str('/home/gideon/Documents/NSP2/LISA data/Verschillende hoogtes/Sky_angles/Sky_angles')
degrees = str('30')
measurement=str('002')

meting = Normalize(data_folder, degrees, measurement)
x, y = meting.isolate(430,445)

xm,ym = meting.mask_peak(438)
ys=meting.smooth_function()
meting.curve_fit()
yn = meting.normalize()

plt.figure()
plt.plot(x,y,'o')
plt.plot(xm,ym,'o')
plt.plot(xm,ys,'o')

plt.figure()
plt.plot(x,yn)

# RECHTHOEK MAKEN
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
element_1 = equivalent_width(spec1, regions=SpectralRegion(6643.0*u.AA,6644*u.AA))
element_2 = equivalent_width(spec1, regions=SpectralRegion(6645.0*u.AA,6645.5*u.AA))
ratio = element_2/element_1

# Rounding
element_1 = np.round(element_1,3)
element_2 = np.round(element_2,3)
ratio = np.round(ratio,3)

# Print the Equivalent width and ratio of abundance
print('EW Ni I 6643A\tEW Eu II 6645 A\tEu/Ni\tName') # Print a header row
print(str(element_1)+'\t'+str(element_2)+'\t'+str(ratio)+'\t'+label1) # Print for target 1
