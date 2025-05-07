import os, signal
import sys
# Assuming this is ran in the root of the europa_sim repo. 
# Add current directory in path
BASE_PATH = os.getcwd() + "/"
print(BASE_PATH)
sys.path.append(BASE_PATH)

import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.transform_utils import img2blender, blender2img
from classes.ortho_image_manager import OrthoImageManager


path = "hirise_assets/jezero_crater/ortho-images/ESP_046060_1985_RED_C_01_ORTHO.JP2"
ortho_width = 6737.48338021085
ortho_height = 14403.857021143112
map_manager = OrthoImageManager(path, [ortho_width, ortho_height])
img  = map_manager.img
map_width = map_manager.img_size[0]
map_height = map_manager.img_size[1]
px_res = map_manager.px_res
map_manager.print_data()


terrain_ortho_width = map_manager.terrain_ortho_size[0]
terrain_ortho_height = map_manager.terrain_ortho_size[1]
print(terrain_ortho_width)
R_bt = map_manager.rot_matrix # rotation matrix from the blender reference frame to terrain reference frame
t_bt = map_manager.center_blend # position of the terrain center in the Blender frame


# # Sample locations on terrain
samples = 4000
margin = 400 # m


yrange_m = (-terrain_ortho_height/2 + margin, terrain_ortho_height/2 - margin)
xrange_m = (-terrain_ortho_width/2 + margin, terrain_ortho_width/2 - margin)
query_locs_x = np.random.uniform(low=xrange_m[0], high=xrange_m[1], size=samples)
query_locs_y = np.random.uniform(low=yrange_m[0], high=yrange_m[1], size=samples)

query_locs_b = np.zeros((samples,2))
for i in range(samples):
    query_loc_m = np.array([query_locs_x[i], query_locs_y[i]])
    query_loc_b = np.dot(R_bt, query_loc_m) + t_bt
    query_locs_b[i] = query_loc_b
    # print(f"i:{i}, query_loc_b = {query_locs_b[i]}")

# Convert the image to RGB format for Matplotlib
output_image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display the result using Matplotlib
fig, ax = plt.subplots()
ax.imshow(output_image_rgb)
for i in range(samples):
    point_blender = query_locs_b[i,:]
    point_px = blender2img(point_blender, [map_width, map_height], px_res)
    ax.plot(point_px[0], point_px[1], marker="o", color='r', markersize='2')
ax.set_title("Sampled locations")
ax.axis('off')  # Hide the axis
plt.show()


# # Find contours in the binary image
# contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Find the largest contour, assuming it's the terrain
# largest_contour = max(contours, key=cv2.contourArea)

# # Edges of the bounding box
# left_edge = 0
# right_edge = image.shape[1]
# top_edge = 0
# bottom_edge = image.shape[0]

# # Initialize variables to hold the closest points
# west_px = np.zeros((1,2), dtype=np.int)
# east_px = np.zeros((1,2), dtype=np.int)
# north_px = np.zeros((1,2), dtype=np.int)
# south_px = np.zeros((1,2), dtype=np.int)

# # Initialize minimum distances with infinity
# min_dist_left = float('inf')
# min_dist_right = float('inf')
# min_dist_top = float('inf')
# min_dist_bottom = float('inf')

# # Find the closest points to each edge
# for point in largest_contour:
#     px, py = point[0]

#     # Check distance to the left edge
#     dist_left = abs(px - left_edge)
#     if dist_left < min_dist_left:
#         min_dist_left = dist_left
#         west_px[0][0] = px
#         west_px[0][1] = py

#     # Check distance to the right edge
#     dist_right = abs(px - right_edge)
#     if dist_right < min_dist_right:
#         min_dist_right = dist_right
#         east_px[0][0] = px
#         east_px[0][1] = py


#     # Check distance to the top edge
#     dist_top = abs(py - top_edge)
#     if dist_top < min_dist_top:
#         min_dist_top = dist_top
#         north_px[0][0] = px
#         north_px[0][1] = py


#     # Check distance to the bottom edge
#     dist_bottom = abs(py - bottom_edge)
#     if dist_bottom < min_dist_bottom:
#         min_dist_bottom = dist_bottom
#         west_px[0][0] = px
#         west_px[0][1] = py



# # Draw the contour and closest points on the original image
# output_image = image.copy()
# cv2.drawContours(output_image, [largest_contour], -1, (0, 255, 0), 2)
# cv2.circle(output_image, tuple(west_px[0]), 5, (255, 0, 0), -1)
# cv2.circle(output_image, tuple(east_px[0]), 5, (0, 0, 255), -1)
# cv2.circle(output_image, tuple(north_px[0]), 5, (0, 255, 0), -1)
# cv2.circle(output_image, tuple(south_px[0]), 5, (0, 255, 255), -1)
# # cv2.circle(output_image, center_px, 50, (255, 255, 255), -1)



