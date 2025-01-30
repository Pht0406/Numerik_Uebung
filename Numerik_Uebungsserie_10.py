import time
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import erf


def trapez_quad(f, a, b, n):
    h = (b - a) / n
    sum = 0
    for i in range(1, n):
        xi = a + i * h
        sum += f(xi)

    integral = h / 2 * (f(a) + 2 * sum + f(b))
    return integral

def gauss_quad(f, a, b, n):
    xi, wi, = leggauss(n)

    x_trans = 0.5 * (b - a) * xi + 0.5 * (b + a)
    w_trans = 0.5 * (b - a) * wi

    integral = 0
    for i in range(len(w_trans)):
        integral += w_trans[i] * f(x_trans[i])
    return integral


def function(x):
    return 2 / np.sqrt(np.pi) * np.exp(-x * x)


def error(exact, numerical):
    return abs(exact - numerical)

a = 0
b = 4
exact_sol = (erf(b) - erf(a))

n_values = [5, 10, 20, 30, 40]
error_list_trapez = []
error_list_gauss = []
time_list_trapez = []
time_list_gauss = []

for n in n_values:
    starttime_trapez = time.time()
    numerical_sol_trapez = trapez_quad(function, a, b, n)
    endtime_trapez = time.time()
    calcerror_trapez = error(exact_sol, numerical_sol_trapez)

    starttime_gauss = time.time()
    numerical_sol_gauss = gauss_quad(function, a, b, n)
    endtime_gauss = time.time()
    calcerror_gauss = error(exact_sol, numerical_sol_gauss)

    error_list_trapez.append(calcerror_trapez)
    time_list_trapez.append(endtime_trapez - starttime_trapez)

    error_list_gauss.append(calcerror_gauss)
    time_list_gauss.append(endtime_gauss - starttime_gauss)
    print(f"n = {n}, Trapez-Fehler = {calcerror_trapez:.10e}, Gauß-Fehler = {calcerror_gauss:.10e}")

print("\nFehlerordnung Trapezsumme:")
for i in range(1, len(n_values)):
    ratio = error_list_trapez[i - 1] / error_list_trapez[i]  # Verhältnis der Fehler
    print(f"n = {n_values[i - 1]} -> n = {n_values[i]}: Fehlerverhältnis = {ratio:.2f}")
print("\nFehlerordnung Gauss:")
for i in range(1, len(n_values)):
    ratio = error_list_gauss[i - 1] / error_list_gauss[i]  # Verhältnis der Fehler
    print(f"n = {n_values[i - 1]} -> n = {n_values[i]}: Fehlerverhältnis = {ratio:.2f}")


plt.figure(figsize=(12, 6))

# Fehlerplot: Trapezregel vs. Gauß-Quadratur
plt.subplot(1, 2, 1)
plt.loglog(n_values, error_list_trapez, marker='o', label='Trapez-Fehler')
plt.loglog(n_values, error_list_gauss, marker='s', label='Gauß-Fehler')
plt.xlabel('n (Teilintervalle)')
plt.ylabel('Fehler')
plt.title('Fehler')
plt.grid(True, which='both', linestyle='--')
plt.legend()

# Zeitplot: Trapezregel vs. Gauß-Quadratur
plt.subplot(1, 2, 2)
plt.plot(n_values, time_list_trapez, marker='o', label='Trapez-Zeit')
plt.plot(n_values, time_list_gauss, marker='s', label='Gauß-Zeit')
plt.xlabel('n (Teilintervalle)')
plt.ylabel('Laufzeit (Sekunden)')
plt.title('Laufzeit')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
