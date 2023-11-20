from neon_lines import a, b, c
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import pathlib
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
        
        file = pathlib.Path(str(data_folder), str(f'{degrees}deg-{measurement}.fit'))
        self.data = fits.getdata(file)
        
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
        
    def mask_peak(self, wavelength, width_multiplier):
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
        
        width = width_multiplier*width
        
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
        

    def smooth_function(self, smoothing):
        
        y = self.y_masked
        self.smooth_y = gaussian_filter1d(y,sigma=smoothing)
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
x, y = meting.isolate(640,660)

xm,ym = meting.mask_peak(656, 0.5)
ys=meting.smooth_function(10)
meting.curve_fit()
yn = meting.normalize()

plt.figure()
plt.plot(x,y,'-o')
plt.plot(xm,ym,'o')
plt.xlabel('golflengte (nm)')
plt.ylabel('relatieve intensiteit')
plt.savefig('mask.png', dpi=300)
# plt.plot(xm,ys,'o')

plt.figure()
plt.plot(x,yn, 'o')
plt.xlabel('golflengte (nm)')
plt.ylabel('relatieve intensiteit')
plt.savefig('norm.png', dpi=300)