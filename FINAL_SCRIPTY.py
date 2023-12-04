import matplotlib.pyplot as plt
from normalize import Normalize
import numpy as np
from absorption_at_diff_angles import angles
from lmfit import Model

def graph_and_fit(wavelength, corr, small_min_corr, small_max_corr):
    
    mini = wavelength - corr
    maxi = wavelength + corr
    small_min = wavelength - small_min_corr
    small_max = wavelength + small_max_corr


    x, y, err = angles(mini, maxi, wavelength, 0.5, 10, small_min, small_max)
    
    def func(x, a, b):
        
        return a * x + b
    
    model = Model(func)
    pars = model.make_params(a=1, b=1)
    
    
    result = model.fit(y, x=x, weights=1/err, params=pars)
    
    a = result.params['a'].value
    print(a)
    
    return x, y, err, result
    

    
wavelength = 630
corr = 10
small_min_corr = 2
small_max_corr = 2

x,y,err,result = graph_and_fit(wavelength, corr, small_min_corr, small_max_corr)

plt.figure()
plt.title(f'area vs degrees {wavelength}')
plt.errorbar(x, y, yerr=err, fmt='o')
plt.plot(x, result.best_fit)
plt.xlabel(r'hoek ($^{\circ}$)')
plt.xlim(0,90)
plt.ylim(0,0.5)
plt.ylabel('oppervlakte spectraallijn')
plt.rcParams['figure.dpi'] = 300
plt.show()

wavelength = 493
corr = 10 
small_min_corr = 1
small_max_corr = 1

x,y,err,result = graph_and_fit(wavelength, corr, small_min_corr, small_max_corr)


plt.errorbar(x, y, yerr=err, fmt='o')
plt.plot(x, result.best_fit)
plt.title(f'Schuine fit {wavelength}')
plt.rcParams['figure.dpi'] = 300
plt.xlim(0,90)
plt.ylim(0,0.5)
plt.show()


wavelength = 656
corr = 10
small_min_corr = 1
small_max_corr = 2
graph_and_fit(wavelength, corr, small_min_corr, small_max_corr)

x,y,err,result = graph_and_fit(wavelength, corr, small_min_corr, small_max_corr)


plt.errorbar(x, y, yerr=err, fmt='o')
plt.plot(x, result.best_fit)
plt.title(f'rechte fit {wavelength}')
plt.rcParams['figure.dpi'] = 300
plt.xlim(0,90)
plt.ylim(0,0.5)
plt.show()