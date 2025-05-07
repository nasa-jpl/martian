import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

def load_depth_data(filepath):
    # Load the depth data from the .npy file
    dmap = np.load(filepath)
    return dmap

def visualize_data_2d(dmap, title="2D Map"):
    # Visualize the depth map using matplotlib
    plt.figure(figsize=(10, 8))
    plt.imshow(dmap, cmap='gray')
    plt.colorbar()
    plt.title(title)
    plt.show(block=False)

def visualize_data_3d(dmap, title="3D Map"):
    # Create a meshgrid of coordinates (x, y) for the depth data
    x = np.linspace(0, dmap.shape[1] - 1, dmap.shape[1])
    y = np.linspace(0, dmap.shape[0] - 1, dmap.shape[0])
    X, Y = np.meshgrid(x, y)
    
    # Create a 3D plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    # Plotting the surface plot
    surf = ax.plot_surface(X, Y, dmap, cmap='viridis', edgecolor='none')
    ax.set_title(title)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Depth')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)  # Add a color bar which maps values to colors.
    plt.show(block=False)





filepath = './visualizations/sample_render/MapCam0000_depth.npy'
cam_altitude = 3000 # m
cam_clip_start = 0.01 # m

# Visualize depth
dmap = load_depth_data(filepath)

# Original depth data:
visualize_data_2d(dmap, "Original Depth Map")


dmap[dmap <= cam_clip_start ] = cam_altitude
dmap_norm = (dmap - np.min(dmap)) / (np.max(dmap) - np.min(dmap)) * 255
visualize_data_2d(dmap, "Depth Map")
visualize_data_2d(dmap_norm, "Normalized Depth Map")
# visualize_data_3d(dmap, "Original 3D Depth Map")



# Visualize elevation
emap = cam_altitude - cam_clip_start - dmap 
emap[emap >= cam_altitude - cam_clip_start] = 0
emap_norm = (emap - np.min(emap)) / (np.max(emap) - np.min(emap)) * 255
# emap = -emap
visualize_data_2d(emap, "Elevation Map")
# visualize_data_3d(emap, "3D elevation Map")
visualize_data_2d(emap_norm, "Normalized Elevation Map")



plt.show()
