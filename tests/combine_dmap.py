import numpy as np
import imageio
import os
import json
import cv2
import glob
import shutil

def load_depth(filepath):
    """
    Load a depth image from a .npy file.

    Args:
        filepath (str): Path to the .npy file (without the extension).

    Returns:
        np.ndarray: Loaded depth image.
    """
    full_path = filepath + ".npy"
    with open(full_path, "rb") as file:
        depth = np.load(file)
    print(f"Depth data loaded from '{full_path}'")
    return depth

store_dir = "/home/dpisanti/LORNA/mars-sim/dataset/roughness_glossy_test_full_res/maps_roughness_0.8_glossy_tiles/elev_40_azim_180"
depth_dir = store_dir + "/depth/"
cam_name = "OrthoCam"
id = 0
tiles_x = 8
tiles_y = 16
scene_name = "Ortho"
cam_data_type = "ORTHO"
cam_clip_start = 100
resolution_x = 26949
resolution_y = 57615


filepath = depth_dir + cam_name + "_%04i" % id



# Combine the tile dmaps
overall_width = resolution_x
overall_height = resolution_y
comb_dmap = np.zeros((overall_height, overall_width), dtype=np.float32)

# # Original solution
# # past each tile into the final depth map
# tile_width_avg = overall_width // tiles_x
# tile_height_avg = overall_height // tiles_y

# for tile_path, x, y in depth_tile_paths:
#     print(f"x: {x}, y: {y}")
#     dmap_tile = load_depth(tile_path)
#     print(dmap_tile.shape)
#     # Dinamically compute tile size to account for the proper paste of the last tiles (that may have different size)
#     tile_width = dmap_tile.shape[1]
#     tile_height = dmap_tile.shape[0]
#     print((tile_width, tile_height))
#     # comb_dmap[(tiles_y - y - 1) * tile_height:(tiles_y - y) * tile_height, x * tile_width:(x + 1) * tile_width] = dmap_tile
#     comb_dmap[(tiles_y - y - 1) * tile_height_avg:(tiles_y - y) * tile_height, x * tile_width_avg:(x + 1) * tile_width] = dmap_tile
#     # # Adjust for the last tile if it has a different size
#     # if x == tiles_x - 1:
#     #     comb_dmap[(tiles_y - y - 1) * tile_height:(tiles_y - y) * tile_height, x * tile_width:] = dmap_tile[:, :overall_width - x * tile_width]
#     # if y == tiles_y - 1:
#     #     comb_dmap[(tiles_y - y - 1) * tile_height:, x * tile_width:(x + 1) * tile_width] = dmap_tile[:overall_height - (tiles_y - y - 1) * tile_height, :]

comb_dmap = np.zeros((overall_height, overall_width), dtype=np.float32)


# Loop through each tile and place it in the combined depth map
current_y = overall_height
for y in range(tiles_y):
    current_x = 0
    for x in range(tiles_x):
        
        # Load the tile
        tile_path = os.path.join(depth_dir,f"OrthoCam_0000_depth_tile_{x}_{y}")
        tile = load_depth(tile_path)
        
        # Determine the position to place the tile in the combined depth map
        tile_height, tile_width = tile.shape
        
        # Calculate the starting coordinates in the combined map
        start_x = current_x
        start_y = current_y - tile_height  # y is inverted
        
        # Print
        print(f"\nx: {x}, y:{y}")
        print(f"current_y ={current_y}, current_x = {current_x}")
        print(f"start_y ={start_y}, start_x = {start_x}")
        print(f"tile_height ={tile_height}, tile_width = {tile_width}")
        

        # Place the tile in the combined depth map
        comb_dmap[start_y: start_y + tile_height, start_x:start_x+tile_width] = tile
        print("Tile patched successfully!")
        print("\n")

        current_x += tile_width
    current_y -= tile_height 

filepath += "_depth"
# Save the combined depth map
with open(filepath + ".npy", "wb") as file:
    np.save(filepath + ".npy", comb_dmap)	


print("***Combined depth map patched successfully!***")
cam_location_z = 3959.15
if cam_data_type == "ORTHO":
    comb_dmap[comb_dmap <= cam_clip_start] = cam_location_z
print(f"Start normalizing")
comb_dmap_norm = (comb_dmap - np.amin(comb_dmap))/np.abs(np.amax(comb_dmap) - np.amin(comb_dmap)) * 255.0 
print(f"Start writing")
cv2.imwrite(filepath + '-norm.png', comb_dmap_norm)
print(f"Normalized depth image saved at '{filepath}-norm.png'")
