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
plt.style.use('seaborn-talk')
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
from astropy.utils.data import download_file
from specutils import Spectrum1D
from specutils import SpectralRegion
from specutils.analysis import equivalent_width
from specutils.analysis import fwhm
from astropy.io import fits
from lmfit.models import Model
from scipy.signal import find_peaks
from pathlib import Path
import Pylance
        

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
#                  facecolor='green', interpolate=True, alpha=0.3)
# # Dark gray area.
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
