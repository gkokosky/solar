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


fig = plt.figure(figsize=(7, 6), layout="constrained")
ax1 = fig.add_subplot(221)
ax1.errorbar(
    x,
    y,
    yerr=err,
    fmt="o",
    color="red",
    capsize=3,
    label=r"628 ($\text{O}_2$)",
)
ax1.plot(x, result.best_fit, color="red")
ax1.set_ylim(0, 0.5)
ax1.set_xticks([0, 15, 30, 45, 60, 75, 90], minor=False)
ax1.set_xticks([5, 10, 20, 25, 35, 40, 50, 55, 65, 70, 80, 85], minor=True)
ax1.set_yticks(np.linspace(0, 0.5, 6), minor=False)
ax1.set_yticks(np.linspace(0, 0.5, 11), minor=True)
ax1.set_title("a", loc="left")
ax1.set_xlabel(r"inclinatie ($^{\circ}$)")
ax1.set_ylabel("equivalente breedte (nm)")
plt.rcParams["figure.dpi"] = 300

wavelength = 656
corr = 30
small_min_corr = 0.8
small_max_corr = 2.2

x, y, err, result = graph_and_fit(
    wavelength, corr, small_min_corr, small_max_corr
)

ax4 = fig.add_subplot(222)
ax4.errorbar(
    x, y, yerr=err, fmt="o", color="blue", capsize=3, label="656 (Hα)"
)
ax4.plot(x, result.best_fit, color="blue")
ax4.set_ylim(0, 0.5)
ax4.set_xticks([0, 15, 30, 45, 60, 75, 90], minor=False)
ax4.set_xticks([5, 10, 20, 25, 35, 40, 50, 55, 65, 70, 80, 85], minor=True)
ax4.set_title("b", loc="left")
ax4.set_yticks(np.linspace(0, 0.5, 6), minor=False)
ax4.set_yticks(np.linspace(0, 0.5, 11), minor=True)
ax4.set_xlabel(r"inclinatie ($^{\circ}$)")
ax4.set_ylabel("equivalente breedte (nm)")
# masked point
ax4.errorbar(30, 0.1939, yerr=0.0124, fmt="o", color="gray", capsize=3)

ax3 = fig.add_subplot(223)
wavelength = 730
corr = 30
small_min_corr = 0.85
small_max_corr = 0.85
x, y, err, result = graph_and_fit(
    wavelength, corr, small_min_corr, small_max_corr
)


ax3.errorbar(
    x, y, yerr=err, fmt="o", color="green", capsize=3, label=r"730 (H$_2$O)"
)

ax3.plot(x, result.best_fit, color="green")
ax3.set_xticks([0, 15, 30, 45, 60, 75, 90], minor=False)
ax3.set_xticks([5, 10, 20, 25, 35, 40, 50, 55, 65, 70, 80, 85], minor=True)
ax3.set_yticks(np.linspace(0, 0.5, 6), minor=False)
ax3.set_yticks(np.linspace(0, 0.5, 11), minor=True)
ax3.set_title("c", loc="left")
ax3.set_ylabel("equivalente breedte (nm)")
ax3.set_xlabel(r"inclinatie ($^{\circ}$)")

ax4 = fig.add_subplot(224)
wavelength = 493
corr = 30
small_min_corr = 2.5
small_max_corr = 2.5

x, y, err, result = graph_and_fit(
    wavelength, corr, small_min_corr, small_max_corr
)

ax4.errorbar(
    x, y, yerr=err, fmt="o", color="purple", capsize=3, label="493 (Fe-I)"
)
ax4.plot(x, result.best_fit, color="purple")
ax4.set_ylim(0, 0.5)
ax4.set_xticks([0, 15, 30, 45, 60, 75, 90], minor=False)
ax4.set_xticks([5, 10, 20, 25, 35, 40, 50, 55, 65, 70, 80, 85], minor=True)
ax4.set_title("d", loc="left")
ax4.set_yticks(np.linspace(0, 0.5, 6), minor=False)
ax4.set_yticks(np.linspace(0, 0.5, 11), minor=True)
ax4.set_xlabel(r"inclinatie ($^{\circ}$)")
ax4.set_ylabel("equivalente breedte (nm)")

plt.show()


fig, (ax1) = plt.subplots()
plt.rcParams["figure.dpi"] = 300


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
ax1.set_xticks([0, 15, 30, 45, 60, 75, 90], minor=False)
ax1.set_xticks([5, 10, 20, 25, 35, 40, 50, 55, 65, 70, 80, 85], minor=True)
ax1.set_ylim(0.5, 1.5)
ax1.set_yticks(np.linspace(0.5, 1.5, 5), minor=False)
ax1.set_yticks(np.linspace(0.5, 1.5, 9), minor=True)
ax1.set_xlabel(r"inclinatie ($^{\circ}$)")
ax1.set_ylabel("equivalente breedte (nm)")
plt.show()

with open("rc.txt", "w") as f:
    f.writelines(rc)
