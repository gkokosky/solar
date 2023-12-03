from normalize import Normalize
from scipy.signal import find_peaks, peak_widths
from lmfit import Model
import matplotlib.pyplot as plt
import numpy as np

class Error:
    """ Class finds error on intensity of normalized absorption spectrum through analyzing baseline noise
    """

    def __init__(self, degrees, measurement, min, max, wavelength, width, smoothing):
        
        meting = Normalize(degrees, measurement)
        meting.isolate(min ,max)
        meting.mask_peak(wavelength, width)
        meting.smooth_function(smoothing)
        meting.curve_fit()
        
        self.measurement = measurement
        
        # converts measurement from int to string for data reading
        for i in range(1,11):
            degrees = f'{degrees}'
            measurement = f'{i}'
            if len(measurement) == 1:
                measurement = f'00{i}'
                self.measurement = f'00{i}'
            elif len(measurement) == 2:
                measurement = f'0{i}'
            else:
                print('huh')
                
        # converts angle from int to str for same purpose
        if degrees == 6:
            degrees = f'0{degrees}'
        else:
            degrees = f'{degrees}'
        
        
        self.width = width
        self.x, self.y = meting.normalize()
        
        self.wavelength = wavelength
        
    def peaks(self, width):
        """
        Finds prominent peaks and masks them so error will be determined based on noise only.
        """
        x = self.x
        y = self.y
        
        peaks, _ = find_peaks(-y + 1, height=0.03)
        widths, _, _, _ = np.array(peak_widths(-y + 1, peaks)) * width
        
        x_peaks = x[peaks]
        x_left = x_peaks - width
        x_right = x_peaks + width
        
        left_indices = []
        right_indices = []
        for i in x_left:
            left_diff = np.abs(x - i)
            left_idx = np.argmin(left_diff)
            left_indices.append(left_idx)
        for i in x_right:
            right_diff = np.abs(x-i)
            right_idx = np.argmin(right_diff)
            right_indices.append(right_idx)
            
        left_idx, right_idx = np.array(left_indices), np.array(right_indices)
        
        for i, j in zip(left_idx, right_idx):
            
            mask_range = np.array(range(i, j+1))
            mask = np.ones_like(x, dtype=bool)
            
            mask[mask_range] = False
            
            self.x_masked = x[mask]
            self.y_masked = y[mask]
            
    def error(self):
        """Fits 2nd-degree polynomial through function and determines error based on residuals.
        """
        
        x = self.x_masked
        y = self.y_masked
        
        def function(x, a, b, c):
            return a * x**2 + b*x + c
        
        model = Model(function)
        pars = model.make_params(a=1,b=1,c=1)
        
        result = model.fit(y, pars,x=x)
        diff = np.abs(result.residual)
        err = np.mean(diff)
        
        return err
            
meting = Error(6,2,640, 680, 656, 0.5, 10)
meting.peaks(0.6)
meting.error()
        
