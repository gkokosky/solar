# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 14:06:46 2023

@author: Femke
"""
from normalize import Normalize
# equivalent width method script
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline
import matplotlib.pyplot as plt
from astropy.io import fits
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
from lmfit.models import GaussianModel
from scipy.signal import find_peaks
from pathlib import Path
        
    
data_folder = str('/home/gideon/Documents/NSP2/LISA data/Verschillende hoogtes/Sky_angles/Sky_angles')
degrees = str('30')
measurement=str('002')

meting = Normalize(data_folder, degrees, measurement)
meting.isolate(0,800)
meting.mask_peak(656, 0.5)
meting.smooth_function(10)
meting.curve_fit()
x, y = meting.normalize()

plt.figure()
plt.plot(x,y,'o')
plt.xlabel('golflengte (nm)')
plt.ylabel('relatieve intensiteit')
plt.savefig('normalized.png', dpi=300)


# isolate further for peak only
def isolate(x, y, min,max):
    """Isolates specific wavelength range, so that only one absorption peak is found. 
    
    Args:
        min (float): lowest wavelength in range to analyze
        max (float): highest wavelength in range to analyze
    """
    
    # finds smallest difference between x and min/max
    min_diff = np.abs(x - min)
    max_diff = np.abs(x - max)
    
    # finds index associated with these values
    min_idx = min_diff.argmin()
    max_idx = max_diff.argmin()
    
    x = x[min_idx:max_idx+1]
    y = y[min_idx:max_idx+1]
    
    return x, y

x,y = isolate(x,y, 0,800)
plt.figure()
plt.plot([640,680], [1,1])
plt.xlim(654,660)
plt.plot(x,y,'o')
plt.xlabel('golflengte (nm)')
plt.ylabel('relatieve intensiteit')
plt.savefig('isolated_peak.png', dpi=300)

# drop points above 1 
def drop(x,y):
    
    y_diff = 1 - y
    
    bool = np.where(y_diff>0)
    y_crop = y[bool]
    x_crop = x[bool]
    return x_crop, y_crop
    
x, y = drop(x,y)
plt.figure() 
plt.plot(x,y,'o')   

def Riemann(x,y):
    pass

def polyfit(x,y):
    
    y = -y + 1
    
    model = GaussianModel()
    pars = model.guess(-y, x=x)
    result = model.fit(-y, pars, x=x)
    
    return result

# RECHTHOEK MAKEN
# er wordt gebruik gemaakt van matplotlib tranforms
# dit werkt nog niet samen met de rest ^

# fig2, ax2 = plt.subplots()
# fig2.suptitle('...', fontsize='24')

# ax2.plot(golflengte_1, relative_intensity_1)

# plt.xlabel('λ (Å)',fontsize='20')
# plt.ylabel('Relative flux', fontsize='20')

# ax2.set_xlim([6643.3,6644])

# # Fill in the area under the curve and overlay a rectangle of equal area.
# # Green area
# trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
# ax2.fill_between(golflengte_1, 1, relative_intensity_1, where=relative_intensity_1 < 1*u.Unit('erg cm-2 s-1 AA-1'),
#                   facecolor='green', interpolate=True, alpha=0.3)
# Dark gray area.
# ax2.add_patch(plt.Rectangle((.29, .045), .358, 0.85, ec='k', fc="k",
#                             alpha=0.3,  transform=ax2.transAxes))

# plt.axhline(y=1,ls='--',color='red',lw=2)

# plt.show()

# # EQUIVALENT WIDTH BEPALEN 
# # hierbij word gebruik gemaakt van de specutils package
# element_1 = equivalent_width(spec1, regions=SpectralRegion(6643.0*u.AA,6644*u.AA))
# element_2 = equivalent_width(spec1, regions=SpectralRegion(6645.0*u.AA,6645.5*u.AA))
# ratio = element_2/element_1

# # Afronding
# element_1 = np.round(element_1,3)
# element_2 = np.round(element_2,3)
# ratio = np.round(ratio,3)

# # Print the Equivalent width and ratio of abundance
# print(str(element_1)+'\t'+str(element_2)+'\t'+str(ratio)+'\t'+label1) # Print for target 1
