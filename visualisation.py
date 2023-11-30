from astropy.io import fits
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import gridspec

data = fits.getdata('Sky_angles/Sky_angles/06deg-001.fit')
hdr = fits.getheader('Sky_angles/Sky_angles/06deg-001.fit')
print(repr(hdr))

prof = np.sum(data, axis=0)[0:1201]

fig, (ax1, ax2) = plt.subplots(2, 1, width_ratios=[2], sharex=False)
ax1.imshow(data, cmap='gray', aspect=0.31)
ax1.set_yticks([])
ax1.set_xlim(0,1200)
ax1.set_xticks([])
ax2.plot(prof, color='black')
ax2.set_yticks([])
plt.tight_layout()
plt.xlabel('pixels')
plt.xlim(0,1200)
plt.tight_layout()

plt.savefig('1dreductie.png', dpi=300)

from neon_lines import pixel_peaks, neon_lines, result
from normalize import Normalize

meting = Normalize('06', '010')
x_0, y_0 = meting.isolate(640,655)
meting.mask_peak(647,0.5)
meting.smooth_function(10)
meting.curve_fit()
x,y = meting.normalize()

plt.figure()
plt.subplot(1,2,1)
plt.plot(pixel_peaks, np.array(neon_lines)/10, 'o', color = 'black')
plt.plot(pixel_peaks, result.best_fit / 10,color='black')
plt.title('a')
plt.xlabel('pixels')
plt.ylabel('golflengte (nm)')

plt.subplot(2,2,2)
plt.plot(x_0,y_0, color='black')
plt.title('b')
plt.xlabel('golflengte (nm)')
plt.ylabel('intensiteit')

plt.subplot(2,2,4)
plt.plot(x,y,color='black')
plt.title('c')
plt.xlabel('golflengte (nm)')
plt.ylabel('intensiteit')
plt.tight_layout()

plt.savefig('neon_en_norm.png', dpi=300)
