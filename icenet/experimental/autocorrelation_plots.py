import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.sin(2*np.pi*x / 12)

x = np.linspace(0, 12, 1000)
y = f(x)
r = np.correlate(y, y, mode="full")

# producing plot a
plt.figure()
plt.plot(x, y)
plt.xlabel("Month")
plt.ylabel("Measurement")
plt.savefig("./experimental/autocorrelation_plot_a.pdf", dpi=300)

# producing plot b
plt.figure()
plt.plot(np.linspace(-12, 12, len(r)), r)
plt.xlabel(r"$\tau$ (months)")
plt.ylabel(r"$R_x(\tau)$")
plt.savefig("./experimental/autocorrelation_plot_b.pdf", dpi=300)
