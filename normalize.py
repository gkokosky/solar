from neon_lines import a, b, c
from astropy.io import fits
from lmfit.models import ExponentialGaussianModel, SkewedGaussianModel, GaussianModel, QuadraticModel, LinearModel
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import gaussian_filter1d

class Normalize:
    
    def __init__(self,degrees,data_folder):
        """ Initializes data from 10 measurements.
        Returns unprocessed dataset.
        """
        self.x = np.array([])
        self.y = np.array([])
        self.smooth_y = np.array([])
        
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
    
    def isolate_peaks(self, min,max):
        
        self.x = self.x[min:max]
        self.y = self.y[min:max]
        
        return self.x, self.y
        
    # def mask_peaks(self):
    #     """ Masks absorption peaks for propper normalization.       
    #     """        
    #     peaks = find_peaks(-1 * self.y)
    #     #peak_w = peak_widths(self.y, peaks)
    #     peaks = peaks[0]
        
    #     self.smooth_x = np.array([])
    #     self.smooth_y = np.array([])
        
    #     # make range of dots to remove
    #     peak_range = []
    #     for i in range(len(peaks)):
    #         peak_range.append(peaks[i] - 3)
    #         peak_range.append(peaks[i] - 2)
    #         peak_range.append(peaks[i] - 1)
    #         peak_range.append(peaks[i])
    #         peak_range.append(peaks[i] +1)
    #         peak_range.append(peaks[i] +2)
    #         peak_range.append(peaks[i] +3)

    #     self.smooth_x = np.delete(self.x, peaks)
    #     self.smooth_y = np.delete(self.y, peaks)
        
    #     return self.smooth_x, self.smooth_y
        
    def smooth_function(self):
        
        y = self.y
        self.smooth_y = gaussian_filter1d(y,sigma=1)
        return self.smooth_y        

    def curve_fit(self):
        
        y = self.smooth_y
        x = self.x
        
        q_model = LinearModel()
        
        pars = q_model.guess(y, x=x)
        result = q_model.fit(y,pars,x=x)
        
        # plt.figure()
        # plt.plot(x,y,'o',markersize=0.5)
        # plt.plot(x,result.best_fit)
        # plt.rcParams['figure.dpi'] = 300
        # plt.xlabel('wavelength (nm)')
        # plt.ylabel('relative intensity')
        # plt.savefig('fit.png')
        
        self.fit = result.best_fit
        return result.best_fit
    
    def normalize(self):
        
        y = self.smooth_y
        norm_function = np.array(self.fit)/np.array(self.y)
        norm_function = np.array(self.y)/np.array(self.fit)
        return norm_function
        
data_folder = str('/home/gideon/Documents/NSP2/LISA data/Verschillende hoogtes/Sky_angles/Sky_angles')
degrees = str('06')

meting = Normalize(degrees, data_folder)
x,y = meting.pixel_to_wavelength()
x, y = meting.isolate_peaks(600, 850)
plt.figure()
plt.plot(x,y,'o')

y = meting.smooth_function()
yf =  meting.curve_fit()
plt.figure()
plt.plot(x,y,'o')
plt.plot(x,yf)

yn = meting.normalize()
plt.figure()
plt.plot(x,yn)
