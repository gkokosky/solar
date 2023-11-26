from equivalent_width_method import Area
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


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
        else:
            print('huh')

        measure = Area(degrees, measurement, min, max, wavelength, width, smoothing)
        # measure.isolate(small_min, small_max)
        x,y = measure.peak()
        area = measure.trap()
        area_list.append(area)
        
    avg = np.mean(area_list)
    err = np.std(area_list) / np.sqrt(10)
    
    return np.array(area_list), avg, err

def angles(min, max, wavelength, width, smoothing, small_min, small_max):
    
    angles = ['06', '10', '15', '30', '40', '50', '60', '70', '80', '90']
    
    area_array = []
    avg_list = []
    err_list = []
    for i in angles:
        area, avg, err = ten_mesurements(i, min, max, wavelength, width, smoothing, small_min, small_max)
        area_array.append(area)
        avg_list.append(avg)
        err_list.append(err)
    
    angles = np.array(angles)
    angles = angles.astype(int)
    avg = np.array(avg_list)
    err = np.array(err_list)
    
    plt.figure()
    plt.errorbar(angles, avg, yerr=err, fmt='o', color='black')
    plt.xticks([i for i in range(0,100,10)])
    plt.xlabel(r'hoek ($^{\circ}$)')
    plt.ylabel('oppervlakte spectrallijn')
    plt.savefig('rareoppervlakte.png', dpi=300)
    
angles(650, 670, 656, 0.5, 10, 0,0)