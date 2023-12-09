import matplotlib.pyplot as plt
from normalize import Normalize
import numpy as np
from absorption_at_diff_angles import angles
from lmfit import Model

rc = []


def graph_and_fit(wavelength, corr, small_min_corr, small_max_corr):
    mini = wavelength - corr
    maxi = wavelength + corr
    small_min = wavelength - small_min_corr
    small_max = wavelength + small_max_corr

    x, y, err = angles(mini, maxi, wavelength, 0.5, 10, small_min, small_max)

    def func(x, a, b):
        return a * x + b

    if wavelength == 656:
        x_mask = np.delete(x, 3)
        y_mask = np.delete(y, 3)
        err_mask = np.delete(err, 3)

    else:
        x_mask = x
        y_mask = y
        err_mask = err

    model = Model(func)
    pars = model.make_params(a=1, b=1)

    result = model.fit(y_mask, x=x_mask, weights=1 / err_mask, params=pars)

    a = result.params["a"].value
    a_err = result.params["a"].stderr

    a = np.format_float_positional(a, 6)
    a_err = np.format_float_positional(a_err, 6)

    # this oxygen line has a very broad range, but the official value is 686 nm
    if wavelength == 695:
        wavelength = 686
    print(f"{wavelength} nm : {a} +- {a_err} \n")
    rc.append(f"{wavelength} nm : {a} +- {a_err} \n")

    return x_mask, y_mask, err_mask, result


wavelength = 628
corr = 10
small_min_corr = 2
small_max_corr = 2

x, y, err, result = graph_and_fit(
    wavelength, corr, small_min_corr, small_max_corr
)
fig, (ax1) = plt.subplots()
ax1.errorbar(
    x,
    y,
    yerr=err,
    fmt="o",
    color="blue",
    capsize=3,
    label=r"628 ($\text{O}_2$)",
)
ax1.plot(x, result.best_fit, color="blue")
ax1.set_ylim(0, 0.5)
ax1.set_xticks([0, 15, 30, 45, 60, 75, 90], minor=False)
ax1.set_xticks([5, 10, 20, 25, 35, 40, 50, 55, 65, 70, 80, 85], minor=True)
ax1.set_yticks(np.linspace(0, 0.5, 6), minor=False)
ax1.set_yticks(np.linspace(0, 0.5, 11), minor=True)
plt.xlabel(r"inclinatie ($^{\circ}$)")
plt.ylabel("equivalente breedte (nm)")
plt.rcParams["figure.dpi"] = 300

wavelength = 656
corr = 30
small_min_corr = 0.8
small_max_corr = 2.2

x, y, err, result = graph_and_fit(
    wavelength, corr, small_min_corr, small_max_corr
)


ax1.errorbar(x, y, yerr=err, fmt="o", color="red", capsize=3, label="656 (Hα)")
ax1.plot(x, result.best_fit, color="red")

# masked point
ax1.errorbar(30, 0.1939, yerr=0.0124, fmt="o", color="gray", capsize=3)


wavelength = 493
corr = 30
small_min_corr = 2.5
small_max_corr = 2.5

x, y, err, result = graph_and_fit(
    wavelength, corr, small_min_corr, small_max_corr
)

ax1.errorbar(
    x, y, yerr=err, fmt="o", color="orange", capsize=3, label="493 (Fe-I)"
)
ax1.plot(x, result.best_fit, color="orange")
ax1.legend(bbox_to_anchor=(1.1, 1.05))
plt.tight_layout()
695
plt.show()


fig, (ax1) = plt.subplots()
plt.xlabel(r"hoek ($^{\circ}$)")
plt.ylabel("oppervlakte spectraallijn")
plt.rcParams["figure.dpi"] = 300

wavelength = 730
corr = 30
small_min_corr = 0.85
small_max_corr = 0.85
x, y, err, result = graph_and_fit(
    wavelength, corr, small_min_corr, small_max_corr
)


ax1.errorbar(
    x, y, yerr=err, fmt="o", color="blue", capsize=3, label=r"730 (H$_2$O)"
)

ax1.plot(x, result.best_fit, color="blue")
wavelength = 695
corr = 50
small_min_corr = 10
small_max_corr = 10
x, y, err, result = graph_and_fit(
    wavelength, corr, small_min_corr, small_max_corr
)


ax1.errorbar(
    x,
    y,
    yerr=err,
    fmt="o",
    color="red",
    capsize=3,
    label=r"686 ($\text{O}_2$)",
)
ax1.plot(x, result.best_fit, color="red")
ax1.legend()
plt.show()

with open("rc.txt", "w") as f:
    f.writelines(rc)
