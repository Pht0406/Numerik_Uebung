import numpy as np
import matplotlib.pyplot as plt

A = np.array([[3, 2], [2, 6]])
b = np.array([2, -6])

def gradientenverfahren(A, b, x0, iterations):
    x = x0
    points = [x]  # Speichert die Punkte der Iterationen
    for _ in range(iterations):
        r = b - np.dot(A, x)
        alpha = np.dot(r, r) / np.dot(r, np.dot(A, r))
        x = x + alpha * r
        points.append(x)
    return np.array(points)


# Visualisierung
def plot_gradientenverfahren(A, b, start_points, iterations):

    x_vals = np.linspace(-5, 5, 100)
    y_vals = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Funktional berechnen
    Z = 0.5 * (3 * X ** 2 + 4 * X * Y + 6 * Y ** 2) - 2 * X + 6 * Y

    plt.contour(X, Y, Z, levels=50, cmap='viridis')

    for x0 in start_points:
        curve = gradientenverfahren(A, b, x0, iterations)
        plt.plot(curve[:, 0], curve[:, 1], marker='o', label=f'Start: {x0}')

    # Plot-Details
    plt.title("Gradientenverfahren")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.legend()
    plt.grid()
    plt.show()


# Parameter
start_points = [np.array([0, 0]), np.array([1, 1]), np.array([-2, -2]), np.array([12/7, -11/7])]
iterations = 5

# Plot erstellen
plot_gradientenverfahren(A, b, start_points, iterations)
