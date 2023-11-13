from neon_lines import params
from astropy.io import fits
from lmfit.models import ExponentialGaussianModel, SkewedGaussianModel, GaussianModel
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

class Normalize:
    
    def __init__(self,degrees,data_folder):
        """ Initializes data from 10 measurements.
        Returns unprocessed dataset.
        """
        self.x = np.array([])
        self.dataset = []
        for i in range(1,10):
            data = fits.getdata(f'{data_folder}/{degrees}deg-00{i}.fit')
            self.dataset.append(data)
        self.dataset.append(fits.getdata(f'{data_folder}/{degrees}deg-010.fit'))
        
    def pixel_to_wavelength(self, pars):
        """ Reduces dataset to 1D array and converts pixels on x-axis
        to wavelength in nm using known wavelengths of neon. 
        """
        reduced_dataset = []
        for j in self.dataset:
            reduced_data = np.sum(j, axis=0)
            reduced_dataset.append(reduced_data)
            
        reduced_dataset = np.array(reduced_dataset)
        x_pixel = np.mean(reduced_dataset, axis=0)
        x_pixel_err = np.std(reduced_dataset, axis=0) / np.sqrt(10)
        
        a = pars[0]
        b = pars[1]
        c = pars[2]

        self.x = a * x_pixel**2 + b * x_pixel + c  
        return(self.x)
    
    def mask_peaks(self):
        """ Masks absorption peaks for propper normalization.       
        """        
        # skip dit nu ff
        peaks = find_peaks(self.x)
        
        
    def curve_fit(self):
        
        x = np.array([i for i in range(len(self.x))])
        y = -1 * self.x
        
        eg_model = SkewedGaussianModel()
        pars = eg_model.guess(y, x=x)
        result = eg_model.fit(y,pars,x=x)
        
        self.fit = result.best_fit
        
        return x, result.best_fit
    
    def normalize(self):
        
        norm_function = self.fit/self.x
        return norm_function
        
data_folder = str('/home/gideon/Documents/NSP2/LISA data/Verschillende hoogtes/Sky_angles/Sky_angles')
degrees = str('200')
meting = Normalize(degrees, data_folder)
wave = meting.pixel_to_wavelength(params)
x, fit = meting.curve_fit()

norm = meting.normalize()

plt.figure()
plt.plot(-1 * wave,'o', markersize=0.5)
plt.plot(x, fit)

plt.figure()
plt.plot(x,norm,'o',markersize=0.5)
plt.ylim(-3,0)