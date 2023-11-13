# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 10:55:18 2023

@author: Femke
"""

# probeersel om pixels om te rekenen naar golflengtes
def pixel_to_wavelength(pixel_position, a, b):
    wavelength = a * pixel_position + b
    return wavelength

# Voorbeeld kalibratiecoëfficiënten (moet worden aangepast met echte waarden)
a = 0.1
b = 500

# Pixelpositie die je wilt omrekenen
pixel_position = 100

# Omrekenen naar golflengte
wavelength = pixel_to_wavelength(pixel_position, a, b)

print(f"Pixelpositie: {pixel_position}")
print(f"Golflengte: {wavelength} nm")

############################
# testest