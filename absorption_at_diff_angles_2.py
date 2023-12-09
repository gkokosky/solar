import numpy as np
from equivalent_width_method import Area
from error import Error
import matplotlib.pyplot as plt


def ten_measurements(
    degrees,
    measurements,
    min,
    max,
    wavelength,
    width,
    smoothing,
    small_min,
    small_max,
):
    degrees = np.full_like(measurements, degrees)
    measurements = np.array([f"{i:03d}" for i in measurements])

    measure = Area(
        degrees,
        measurements,
        min,
        max,
        wavelength,
        width,
        smoothing,
        small_min,
        small_max,
    )
    x, y = measure.peak()

    area = measure.trap()

    err_meting = Error(
        degrees, measurements, min, max, wavelength, width, smoothing
    )
    err_meting.peaks(0.5)
    point_err = err_meting.error()

    sigma_list = point_err * np.sqrt(y)
    sigma_list = sigma_list**2

    err = np.sqrt(np.sum(sigma_list)) * (x[1] - x[0])

    return area, x, y, err


def angles(min, max, wavelength, width, smoothing, small_min, small_max):
    angles = np.array([6, 10, 15, 30, 40, 50, 60, 70, 80, 90])

    area_array = []
    avg_list = []
    err_list = []
    for angle in angles:
        measurements = np.arange(1, 11)
        area, x, y, err = ten_measurements(
            angle,
            measurements,
            min,
            max,
            wavelength,
            width,
            smoothing,
            small_min,
            small_max,
        )

        # if angle == 6:
        #     plt.figure()
        #     plt.plot(x, y)
        #     plt.title(f"{wavelength} nm")

        area_array.append(area)

    angles = angles.astype(int)
    avg = np.mean(area_array, axis=1)
    err = np.std(area_array, axis=1)

    return angles, avg, err


# Example usage:
min_val = 0
max_val = 1
wavelength_val = 500
width_val = 1
smoothing_val = 0.5
small_min_val = 0
small_max_val = 10

angles_arr, avg_arr, err_arr = angles(
    min_val,
    max_val,
    wavelength_val,
    width_val,
    smoothing_val,
    small_min_val,
    small_max_val,
)
