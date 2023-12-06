from equivalent_width_method import Area
from error import Error
import matplotlib.pyplot as plt
import numpy as np


# returns area from 10 measurements at 1 angle
def ten_mesurements(
    degrees, min, max, wavelength, width, smoothing, small_min, small_max
):
    area_list = []
    err_list = []
    for i in range(1, 11):
        degrees = f"{degrees}"
        measurement = f"{i}"
        if len(measurement) == 1:
            measurement = f"00{i}"
        elif len(measurement) == 2:
            measurement = f"0{i}"
        else:
            print("huh")

        measure = Area(
            degrees,
            measurement,
            min,
            max,
            wavelength,
            width,
            smoothing,
            small_min,
            small_max,
        )
        x, y = measure.peak()

        if i == 2:
            x_1, y_1 = x, y

        area = measure.trap()
        area_list.append(area)

        err_meting = Error(
            degrees, measurement, min, max, wavelength, width, smoothing
        )
        err_meting.peaks(0.5)
        point_err = err_meting.error()

        # error on point scales with the square root of the flux
        sigma_list = [point_err * np.sqrt(y[i]) for i in range(len(x))]
        sigma_list = (np.array(sigma_list)) ** 2

        # error on area follows from error propagation, multiplied with distance between datapoints
        err = np.sqrt(np.sum(sigma_list)) * (x[1] - x[0])
        err_list.append(err)

    area_list = np.array(area_list)
    avg = np.mean(area_list)
    err = np.mean(err_list)

    return np.array(area_list), avg, err, x_1, y_1


def angles(min, max, wavelength, width, smoothing, small_min, small_max):
    angles = np.array(
        ["06", "10", "15", "30", "40", "50", "60", "70", "80", "90"]
    )

    area_array = []
    avg_list = []
    err_list = []
    for i in angles:
        area, avg, err, x, y = ten_mesurements(
            i, min, max, wavelength, width, smoothing, small_min, small_max
        )

        if i == "06":
            plt.figure()
            plt.plot(x, y)
            plt.title(f"{wavelength} nm")

        area_array.append(area)
        avg_list.append(avg)
        err_list.append(err)

    angles = angles.astype(int)
    avg = np.array(avg_list)
    err = np.array(err_list)

    return angles, avg, err
