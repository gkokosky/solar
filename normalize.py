from neon_lines import a, b, c
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
        self.y = np.array([])
        
        self.dataset = []
        for i in range(1,10):
            data = fits.getdata(f'{data_folder}/{degrees}deg-00{i}.fit')
            self.dataset.append(data)
        self.dataset.append(fits.getdata(f'{data_folder}/{degrees}deg-010.fit'))
        
        self.fit = np.array([])

    def pixel_to_wavelength(self):
        """ Reduces dataset to 1D array and converts pixels on x-axis
        to wavelength in nm using known wavelengths of neon. 
        """
        reduced_dataset = []
        for j in self.dataset:
            reduced_data = np.sum(j, axis=0)
            reduced_dataset.append(reduced_data)
            
        reduced_dataset = np.array(reduced_dataset)
        y_pixel = np.mean(reduced_dataset, axis=0)
        x_pixel = np.array([i for i in range(len(y_pixel))])

        self.x = a * x_pixel**2 + b * x_pixel + c
        self.y = np.array(y_pixel)
        return(self.x, self.y)
    
    def mask_peaks(self):
        """ Masks absorption peaks for propper normalization.       
        """        
        peaks = find_peaks(-1 * self.y)
        self.smooth_x = np.array([])
        self.smooth_y = np.array([])
        self.smooth_x = np.delete(self.x, peaks[0])
        self.smooth_y = np.delete(self.y, peaks[0])
        
        return self.smooth_x, self.smooth_y
        
        
    def curve_fit(self):
        
        y = self.smooth_y
        x = self.smooth_x 
        eg_model = SkewedGaussianModel()
        pars = eg_model.guess(y, x=x)
        result = eg_model.fit(y,pars,x=x)
        
        plt.figure()
        plt.plot(x,y,'o',markersize=0.5)
        plt.plot(x,result.best_fit)
        
        self.fit = result.best_fit
        
        return result.best_fit
    
    def normalize(self):
        
        x = self.smooth_x
        norm_function = np.array(self.fit)/np.array(self.smooth_y)
        return x, norm_function
        
data_folder = str('/home/gideon/Documents/NSP2/LISA data/Verschillende hoogtes/Sky_angles/Sky_angles')
degrees = str('200')
meting = Normalize(degrees, data_folder)
x,y = meting.pixel_to_wavelength()
xs, ys = meting.mask_peaks()

casddasd = meting.curve_fit()
xn, yn = meting.normalize()


plt.figure()
plt.plot(x, y)

plt.figure()
plt.plot(xn,-1*yn+1)
