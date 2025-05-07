import numpy as np
import matplotlib.pyplot as plt

# Define the function to calculate dz/dz_d
def dz_dzd(z, n, f, d):
    S = 2**d - 1  # Bit depth
    return z**2 * ((1/n) - (1/f)) / S

# Plotting dz/dz_d vs z for a specific bit depth, far range, and different near ranges
def plot_dz_dzd_vs_z(d, f, n_values, z_range):
    z_vals = np.linspace(*z_range, 500)  # Create a range of z values
    plt.figure(figsize=(10, 6))
    
    for n in n_values:
        dz_dzd_vals = dz_dzd(z_vals, n, f, d)
        plt.plot(z_vals, dz_dzd_vals, label=f'n = {n} m')
    
    plt.title(f'd (bit depth): {d}, f (far plane distance): {f} m')
    plt.xlabel(r'$z_c$ (Distance from Camera) [m]', fontsize=14)
    plt.ylabel(r'$\frac{dz_c}{dz_d}$ (Precision) [m]', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend(fontsize=14)
    plt.grid(True)

    # Set y-axis to scientific notation
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.xlim(z_range)

    plt.show()

# Parameters
bit_depth = 32  # 32-bit depth
far_plane = 100000 # Fixed far range
near_planes = [0.5, 1, 10] # Different near plane values
z_range = (1, 15000)  # Range of z (distance from camera)

# Plotting
plot_dz_dzd_vs_z(bit_depth, far_plane, near_planes, z_range)
