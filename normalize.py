from neon_lines import params
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

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
      pass      
data_folder = str('/home/gideon/Documents/NSP2/LISA data/Verschillende hoogtes/Sky_angles/Sky_angles')
degrees = str('200')
meting = Normalize(degrees, data_folder)
wave = meting.pixel_to_wavelength(params)

plt.figure()
plt.plot(wave,'o', markersize=0.5)