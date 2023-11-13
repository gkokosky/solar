# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 10:52:57 2023

@author: Femke
"""
# probeersel om kalibratie-coëfficiënten te vinden
import numpy as np
import matplotlib.pyplot as plt

# Voorbeeld van bekende golflengtes en bijbehorende pixelposities
known_wavelengths = np.array([500, 550, 600, 650, 700])  # bijbehorende golflengtes in nm
pixel_positions = np.array([100, 150, 200, 250, 300])  # bijbehorende pixelposities

# Voer lineaire regressie uit om a en b te bepalen
coefficients = np.polyfit(pixel_positions, known_wavelengths, 1)

# De resulterende coëfficiënten zijn a en b
a, b = coefficients

# Plot de gegevens en de beste passende lijn
plt.scatter(pixel_positions, known_wavelengths, color='blue', label='Bekende golflengtes')
plt.plot(pixel_positions, a * pixel_positions + b, color='red', label='Beste passende lijn')
plt.xlabel('Pixelpositie')
plt.ylabel('Golflengte (nm)')
plt.legend()
plt.show()

# Toon de gevonden coëfficiënten
print(f"a: {a}, b: {b}")
