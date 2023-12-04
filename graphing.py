import matplotlib.pyplot as plt
from normalize import Normalize
from absorption_at_diff_angles import ten_mesurements, angles


wavelength = 493
min = wavelength -10
max = wavelength + 10
small_min = wavelength - 2
small_max = wavelength + 2

# norm = Normalize(10,1)
# norm_x, norm_y = norm.isolate(wavelength - 20,wavelength + 20)
# plt.figure()
# plt.plot(norm_x,norm_y)



angles, avg, err = angles(min, max, wavelength, 0.5, 10, small_min, small_max)
plt.figure()
plt.errorbar(angles, avg, yerr=err, fmt='o', color='black', capsize=3.5)
plt.ylim(0,0.5)
plt.xticks([i for i in range(0,100,10)])
plt.xlabel(r'hoek ($^{\circ}$)')
plt.ylabel('oppervlakte spectraallijn')
plt.rcParams['figure.dpi'] = 300
plt.savefig(f'GOED_Fe_{wavelength}_grafiek.png')
plt.show()