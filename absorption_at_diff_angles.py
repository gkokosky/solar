from equivalent_width_method import Area
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

angles = ['06', '10', '15', '30', '40', '50', '60', '70', '80', '90']

# returns area from 10 measurements at 1 angle
def ten_mesurements(degrees, min, max, wavelength, width, smoothing, small_min, small_max):
    
    area_list = []
    for i in range(1,11):
        degrees = f'{degrees}'
        measurement = f'{i}'
        if len(measurement) == 1:
            measurement = f'00{i}'
        elif len(measurement) == 2:
            measurement = f'0{i}'

        measure = Area(degrees, measurement, min, max, wavelength, width, smoothing)
        measure.isolate(small_min, small_max)
        x,y = measure.drop()
        
        area = measure.trap()
        area_list.append(area)
        
    avg = np.mean(area_list)
    err = np.std(area_list) / np.sqrt(10)
    
    return area_list, avg, err
