import numpy as np
import matplotlib.pyplot as plt

map_res = 0.25 # map resoution [m / pixels]
altitude0_min = 64 # min obs altitude [m] in the first altitude range
altitude0_max = 132  # max obs altitude [m]
altitude1_min = altitude0_max # min obs altitude [m] in the second altitude range
altitude1_max = 200  # max obs altitude [m]

# min_map_win_size=640
# max_map_win_size=1000
# map_win_size = np.linspace(min_map_win_size, max_map_win_size, 6)
map_win_sizes = [(1320, 2000), (1980, 3000)] #, (1320, 1980), (1320, 3000)]


sensor_width = 80
focal_length = 32

altitudes0 = np.linspace(altitude0_min, altitude0_max, 50)
altitudes1 = np.linspace(altitude1_min, altitude1_max, 50)

# fig, ax = plt.subplots()
# for i in range(len(map_win_size)):
#     scale_factors1 = map_res*map_win_size[i]/(sensor_width * altitudes1/ focal_length)
#     ax.plot(altitudes1, scale_factors1, label=f"Alt. range 1 - Map win. size: {map_win_size[i]} pixels")
# for i in range(len(map_win_size)):
#     scale_factors2 = map_res*map_win_size[i]/(sensor_width * altitudes2/ focal_length)
#     ax.plot(altitudes2, scale_factors2, label=f"Alt. range 2 - Map win. size: {map_win_size[i]} pixels")
fig, ax = plt.subplots()
for i in range(len(map_win_sizes)):
    scale_factors0 = map_res*map_win_sizes[i][0]/(sensor_width * altitudes0/ focal_length)
    scale_factors1 = map_res*map_win_sizes[i][1]/(sensor_width * altitudes1/ focal_length)
    ax.plot(np.concatenate((altitudes0, altitudes1)), np.concatenate((scale_factors0, scale_factors1)), label=f"Map win. size - Range 1: {map_win_sizes[i][0]} pxl, Range 2: {map_win_sizes[i][1]} pxl")

    
ax.set_xlabel("Obs. altitude [m]", fontsize=16)
ax.set_ylabel("Obs / Map scale factor", fontsize=16)
ax.legend(fontsize=20)
ax.grid()
ax.tick_params(axis='x', which='major', labelsize=16)
ax.tick_params(axis='y', which='major', labelsize=16)
plt.show()