# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 14:06:46 2023

@author: Femke
"""
# equivalent width method script
# NumPy is a common library for handling mathy things.
import numpy as np

# SciPy allows for things like interpolation and curve fitting.
from scipy.interpolate import make_interp_spline, BSpline

# MatPlotLib is the most common way to visualize data in Python.
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
from astropy.utils.data import download_file
from specutils import Spectrum1D
from specutils import SpectralRegion
from specutils.analysis import equivalent_width
from specutils.analysis import fwhm

# DATA IMPORTEREN


# SPECTRUM PLOT
def interp(w, f):
  wInterp = np.linspace(w.min(),w.max(), 300)
  spl = make_interp_spline(w, f)
  fInterp = spl(wInterp)
  return wInterp, fInterp

# Interpolate the data for smoother plots
wave1, flux1 = interp(wave1,flux1)

# Add units to the fluxes and wavelengths
flux1 = flux1*u.Unit('erg cm-2 s-1 AA-1')
wave1 = wave1*u.AA
fig, ax = plt.subplots()
fig.suptitle('Eu II Absorption Detection', fontsize='24')

# Add the other plots here
ax.plot(wave1, flux1, label=label1)

# This displays 2 lines to mark Ni I and Eu II line locations.
plt.axvline(x=6645.127,ls=':')
plt.axvline(x=6643.638,ls=':')

# This labels the x-axis and y-axis
plt.xlabel('λ (Å)',fontsize='20')
plt.ylabel('Relative flux', fontsize='20')

# Display a grid
plt.grid(True)

# Turn on the legend.
ax.legend(loc='best')

# Display all the things we've setup.
plt.show()




















# Rechthoek maken
# Plot one of our stars and annotate the equivalent width.
fig2, ax2 = plt.subplots()
fig2.suptitle('Ni I Equivalent Width', fontsize='24')

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

# Print the EW or Ni I, Eu II, and their ratio.
print('EW Ni I 6643A\tEW Eu II 6645 A\tEu/Ni\tName') # Print a header row
print(str(ni1)+'\t'+str(eu1)+'\t'+str(r1)+'\t'+label1) # Print for target 1
