import numpy as np
import matplotlib.pyplot as plt

### 🏆 FFT-Implementation (rekursiv)
def fft_recursive(f):
    n = len(f)
    if n == 1:
        return f
    else:
        w_n = np.exp(-2j * np.pi / n)
        w = 1
        f_even = fft_recursive(f[::2])
        f_odd = fft_recursive(f[1::2])
        c = np.zeros(n, dtype=complex)
        for k in range(n // 2):
            c[k] = f_even[k] + w * f_odd[k]
            c[k + n // 2] = f_even[k] - w * f_odd[k]
            w *= w_n
        return c

### 🏆 IFFT-Implementation (rekursiv)
def ifft_recursive(c):
    n = len(c)
    if n == 1:
        return c
    else:
        w_n = np.exp(2j * np.pi / n)  # Inverse FFT nutzt e^(2πi/n)
        w = 1
        c_even = ifft_recursive(c[::2])
        c_odd = ifft_recursive(c[1::2])
        f = np.zeros(n, dtype=complex)
        for k in range(n // 2):
            f[k] = c_even[k] + w * c_odd[k]
            f[k + n // 2] = c_even[k] - w * c_odd[k]
            w *= w_n
        return f

### 📌 Erzeugung der Treppenfunktion
n = 32
x = np.linspace(0, 2 * np.pi, n, endpoint=False)
f_values = np.where(x < np.pi, 0, 1)  # Treppenfunktion

# 🚀 Berechnung der Fourier-Koeffizienten
c = fft_recursive(f_values)

# 🚀 Verlängerung des Koeffizientenvektors (Zero-Padding für mehr Punkte)
m = 8 * n
c_extended = np.zeros(m, dtype=complex)
c_extended[:n//2] = c[:n//2]  # Positive Frequenzen beibehalten
c_extended[-(n//2):] = c[-(n//2):]  # Negative Frequenzen einfügen

# 🚀 Rücktransformation mit erhöhter Punktanzahl
f_reconstructed = ifft_recursive(c_extended).real  # Nur Realteil
f_reconstructed /= n  # Skalierungskorrektur (NumPy-Standard)

### 📈 Korrektes Plotten der Treppenfunktion & Fourier-Rekonstruktion
x_fine = np.linspace(0, 2 * np.pi, m, endpoint=False)
plt.figure(figsize=(8, 5))
plt.plot(x_fine, f_reconstructed, label="Fourier-Rekonstruktion")
plt.step(x, f_values, label="Originale Treppenfunktion", linestyle='--', where='post', color='orange')
plt.legend()
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Fourier-Approximation der Treppenfunktion")
plt.grid()
plt.show()
