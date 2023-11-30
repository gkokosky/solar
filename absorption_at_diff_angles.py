from equivalent_width_method import Area
from pathlib import Path
import matplotlib.pyplot as plt
import uncertainties
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

        measure = Area(degrees, measurement, min, max, wavelength, width, smoothing, small_min, small_max)
        # measure.isolate(small_min, small_max)
        x,y = measure.peak()
        area = measure.trap()
        area_list.append(area)
        
        # y_with_err = []
        # for i in range(len(y)):
            
        #     err = 0.02
        #     y_err = uncertainties.ufloat(y[i], err)
        #     y_with_err.append(y_err)
        
        # #determine area with error
        # err_arr = np.array(y_with_err)
        # delta_x = x[1] - x[0]
        # print(np.sum(err_arr))
        # err = 0.5 * (err_arr) * delta_x
            
            
        
    area_list = np.array(area_list)
    avg = np.mean(area_list)
    

    return np.array(area_list), avg

def angles(min, max, wavelength, width, smoothing, small_min, small_max):
    
    angles = ['06', '10', '15', '30', '40', '50', '60', '70', '80', '90']
    
    area_array = []
    avg_list = []
    err_list = []
    for i in angles:
        area, avg = ten_mesurements(i, min, max, wavelength, width, smoothing, small_min, small_max)
        area_array.append(area)
        avg_list.append(avg)
    
    angles = np.array(angles)
    angles = angles.astype(int)
    avg = np.array(avg_list)
    err = np.array(err_list)
    
    plt.figure()
    plt.plot(angles, avg, 'o', color='black')
    plt.xticks([i for i in range(0,100,10)])
    plt.xlabel(r'hoek ($^{\circ}$)')
    plt.ylabel('oppervlakte spectraallijn')
    plt.savefig('neon_mooi.png', dpi=300)
    plt.show()
    
angles(695, 710, 703, 0.5, 10, 702.5, 704)