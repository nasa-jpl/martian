import numpy as np
import imageio
import os
import json
import cv2
import glob
import shutil
import bpy
from PIL import Image


# def load_depth(path):
#     # Using imageioi
#     img = imageio.v2.imread(path, format='EXR-FI')
#     dmap = img[..., 0]
#     # # Using opencv
#     # img = cv2.imread(f"./tmp/{scene.name}_depth0000.exr", cv2.IMREAD_UNCHANGED)
#     # dmap = img[..., 2]
#     dmap[dmap>10000] = 0 # Remove points that are far away
#     # print(f"{scene.name}: dmap (min, max) = ({np.amin(dmap)},{np.max(dmap)})")
#     return dmap



def normalize_depth(dmap, cam):
    if cam.data.type == "ORTHO":
        dmap[dmap <= cam.data.clip_start] = cam.location[2]
    dmap_norm = (dmap - np.amin(dmap))/np.abs(np.amax(dmap) - np.amin(dmap)) * 255.0 
    return dmap_norm


################## Data saving utilities ###########################################

def write_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

# def save_elevation(scene, cam, store_dir, id, save_png = False, apply_normalization=False):
    
#     if not os.path.isdir(store_dir):
#             os.makedirs(store_dir)
#     filepath = store_dir + "/" + cam.name + "%04i" % id

#     # print(str(cam.name))
#     # print(str(str(cam.name).replace("Cam","")))
#     img = imageio.v2.imread(f"./tmp/{scene.name}_depth0001.exr", format='EXR-FI')
#     dmap = img[..., 0]
#     dmap[dmap>10000] = 0 # Remove points that are far away

#     filepath += "_elevation"
#     emap = cam.location[2] - cam.data.clip_start - dmap 

#     with open(filepath + ".npy", "wb") as file:
#         np.save(filepath + ".npy", emap)	
#     print(f"Elevation data written in '{filepath}.npy'")

#     if save_png:
#         if "Map" in cam.name:
#             emap[emap >= cam.location[2] - cam.data.clip_start] = 0
#         # cv2.imwrite(filepath + '.png', emap)
#         # print(f"Elevation image saved at '{filepath}.png'")
#         if apply_normalization:
#             emap_norm = (emap - np.amin(emap))/np.abs(np.amax(emap) - np.amin(emap)) * 255.0 
#             cv2.imwrite(filepath + '-norm.png', emap_norm)
#             print(f"Normalized elevation image saved at '{filepath}-norm.png'")

def save_depth(scene, cam, store_dir, id, save_png = False):
    
    depth_dir = store_dir + "/depth/"
    if not os.path.isdir(depth_dir):
            os.makedirs(depth_dir)
    filepath = depth_dir + cam.name + "_%04i" % id

    # Find the .EXR files in the .EXR directory and rename them
    exr_dir = scene.node_tree.nodes['File Output'].base_path
    file_pattern = os.path.join(exr_dir, f"{scene.name}_depth*.exr") 
    files = glob.glob(file_pattern) # Find the file that matches the pattern
    if files: # Check if any file is found
        # Rename the first matching file
        os.rename(files[0], os.path.join(exr_dir, f"{scene.name}_depth.exr"))

    # # Comment the row below if image cannot load .exr file
    # img = imageio.v2.imread(f"{exr_dir}/{scene.name}_depth.exr", format='EXR-FI')

    # Uncomment the block below to load .exr file with opencv
    exr_path = f"{exr_dir}/{scene.name}_depth.exr"
    img = cv2.imread(exr_path, cv2.IMREAD_UNCHANGED)
    # Check if the image was successfully loaded
    if img is None:
        raise FileNotFoundError(f"Failed to read EXR file at {exr_path}")
    
    dmap = img[..., 0]
    dmap[dmap>10000] = 0 # Remove points that are far away
    print(f"Depth shape: {dmap.shape}")
    filepath += "_depth"
    with open(filepath + ".npy", "wb") as file:
        np.save(filepath + ".npy", dmap)	
    print(f"Depth data written in '{filepath}.npy'")

    if save_png:
        if cam.data.type == "ORTHO":
            dmap[dmap <= cam.data.clip_start] = cam.location[2]
        dmap_norm = (dmap - np.amin(dmap))/np.abs(np.amax(dmap) - np.amin(dmap)) * 255.0 
        cv2.imwrite(filepath + '-norm.png', dmap_norm)
        print(f"Normalized depth image saved at '{filepath}-norm.png'")

    # Remove the .exr file
    file_path = os.path.join(exr_dir, f"{scene.name}_depth.exr")
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print(f'Failed to delete {file_path}. Reason: {e}')


def save_depth_tiles(scene, cam, store_dir, id, tiles_x, tiles_y, save_png = False):
    
    depth_dir = store_dir + "/depth/"
    if not os.path.isdir(depth_dir):
            os.makedirs(depth_dir)
    filepath = depth_dir + cam.name + "_%04i" % id

    # Rename tile depth files
    exr_dir = scene.node_tree.nodes['File Output'].base_path

    for y in range(tiles_y):
        for x in range(tiles_x):
            file_pattern = os.path.join(exr_dir, f"{scene.name}_depth_tile_{x}_{y}_*.exr")

            # Find the file that matches the pattern
            files = glob.glob(file_pattern)
            print(files)

            # Check if any file is found
            if files:
                # Rename the first matching file
                os.rename(files[0], os.path.join(exr_dir, f"{scene.name}_depth_tile_{x}_{y}.exr"))
            
            # # Comment the row below if image cannot load .exr file
            # img = imageio.v2.imread(f"{exr_dir}/{scene.name}_depth_tile_{x}_{y}.exr", format='EXR-FI')
                             
            # Uncomment the block below to load .exr file wtih opencv
            exr_path = f"{exr_dir}/{scene.name}_depth_tile_{x}_{y}.exr"
            img = cv2.imread(exr_path, cv2.IMREAD_UNCHANGED)
            # Check if the image was successfully loaded
            if img is None:
                raise FileNotFoundError(f"Failed to read EXR file at {exr_path}")
            
            dmap = img[..., 0]
            dmap[dmap>10000] = 0 # Remove points that are far away
            print(f"Depth shape: {dmap.shape}")
            tile_filepath = filepath + f"_depth_tile_{x}_{y}"
            with open(tile_filepath + ".npy", "wb") as file:
                np.save(tile_filepath + ".npy", dmap)	
            print(f"Tile Depth data written in '{tile_filepath}.npy'")

            # Save tile depth png
            if save_png:
                if cam.data.type == "ORTHO":
                    dmap[dmap <= cam.data.clip_start] = cam.location[2]
                dmap_norm = (dmap - np.amin(dmap))/np.abs(np.amax(dmap) - np.amin(dmap)) * 255.0 
                cv2.imwrite(tile_filepath + '-norm.png', dmap_norm)
                print(f"Normalized tile depth image saved at '{tile_filepath}-norm.png'")

            # Remove the .exr file
            file_path = os.path.join(exr_dir, f"{scene.name}_depth_{x}_{y}.exr")
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
                  
    try:
        # Combine the tile dmaps
        overall_width = scene.render.resolution_x
        overall_height = scene.render.resolution_y
        comb_dmap = np.zeros((overall_height, overall_width), dtype=np.float32)

        # Loop through each tile and place it in the combined depth map
        current_y = overall_height
        for y in range(tiles_y):
            current_x = 0
            for x in range(tiles_x):
                
                # Load the tile
                tile_path = filepath + f"_depth_tile_{x}_{y}"
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
        print(f"Combined Depth data written in '{filepath}.npy'")

        # Save the normalized combined depth in .png format
        if save_png:
            if cam.data.type == "ORTHO":
                comb_dmap[comb_dmap <= cam.data.clip_start] = cam.location[2]
            comb_dmap_norm = (comb_dmap - np.amin(comb_dmap))/np.abs(np.amax(comb_dmap) - np.amin(comb_dmap)) * 255.0 
            cv2.imwrite(filepath + '-norm.png', comb_dmap_norm)
            print(f"Normalized combined depth image saved at '{filepath}-norm.png'")
    except Exception as e:
        print(f"Error in combining the depth tiles: {e}")


def save_cam_data(cam_obj, scene_obj, store_dir):
    
    if not os.path.isdir(store_dir):
            os.makedirs(store_dir)
    filepath = store_dir + "/cam_data.json" 
    data = {
        "name":cam_obj.name,

        # Camera object properties
        "clip_start":cam_obj.data.clip_start,
        "clip_end":cam_obj.data.clip_end,

        # Image properties
        "img_width":scene_obj.render.resolution_x,    # Image width (pixels)
        "img_height":scene_obj.render.resolution_y,  # Image height (pixels)
        "aspect_x":scene_obj.render.pixel_aspect_x,      # pixel scaling factor for width
        "aspect_y":scene_obj.render.pixel_aspect_y,      # pixel scaling factor for height
    }

    # Camera object perspective / ortho properties
    if cam_obj.data.type == "PERSP":
        data["proj_model"]=cam_obj.data.type
        data["focal_length"]=cam_obj.data.lens
        data["sensor_width"]=cam_obj.data.sensor_width
        data["sensor_fit"]=cam_obj.data.sensor_fit
        data["shift_x"]=cam_obj.data.shift_x
        data["shift_y"]=cam_obj.data.shift_y
        # Compute and save intriniscs
        image_aspect = data["aspect_x"] / data["aspect_y"] # Image aspect = aspect_x / aspect_y = fx / fy 
        data["fx"] = (data["img_width"]/data["sensor_width"])*data["focal_length"] # Focal distance x (pixels).
        data["fy"] = image_aspect*data["fx"]    # Focal distance y (pixels).
        data["cx"] = data["img_width"]/2.       # Principal point x (pixels).
        data["cy"] = data["img_height"]/2.      # Principal point y (pixels).
    elif cam_obj.data.type == 'ORTHO':
        data["proj_model"]=cam_obj.data.type
        data["sensor_fit"]=cam_obj.data.sensor_fit
        data["ortho_scale"]=cam_obj.data.ortho_scale
    else:
        raise("Projective model not supported. Select 'PERSP' or 'ORTHO'.")
    write_json(data, filepath)

def save_pose_data(cam, altitude, store_dir, id):
    poses_dir = store_dir + "/cam_poses/"
    if not os.path.isdir(poses_dir):
            os.makedirs(poses_dir)
    filepath = poses_dir + cam.name + "_%04i" % id
    yaw, pitch, roll = cam.rotation_euler[2], cam.rotation_euler[0], cam.rotation_euler[1] # Yaw, pitch, roll (rad) in blender frame
    data = {
        "t_wc_w":list(cam.location), #Vector from camera orgin to world (i.e. Blender) frame origin in w coordinates
        "ypr":[yaw, pitch, roll],    # Euler angles (yaw, pitch, roll) in the Blenderreference frame (deg)
        "altitude": altitude                 # Altitude
    }
    write_json(data, filepath + '_pose.json')


def save_config_from_blender(scene_obj, light_obj, store_dir):
    if not os.path.isdir(store_dir):
            os.makedirs(store_dir)
    filepath = store_dir + "/config_from_blender.json" 
   
    terrain_data ={
        "high_res_ortho":bpy.data.images[0].name,
        "roughness": bpy.data.materials["Mars"].node_tree.nodes["Principled BSDF"].inputs[2].default_value
    }

    light_data={
        "translation":[light_obj.location[0], light_obj.location[1], light_obj.location[2]], # m
        "rotation":[light_obj.rotation_euler[0]*180/np.pi, light_obj.rotation_euler[0]*180/np.pi, light_obj.rotation_euler[0]*180/np.pi], # deg
        "energy":light_obj.data.energy, # W / m2
        "angular_diameter":light_obj.data.angle*180/np.pi, # deg
        "glossy":light_obj.visible_glossy
    }

    world_data={
        "exposure":scene_obj.view_settings.exposure
    }

    
    cycles = scene_obj.cycles
    cycles_data={
        "samples":cycles.samples,
        "device": cycles.device,
        "use_adaptive_sampling": cycles.use_adaptive_sampling,
        # "use_progressive_refine": cycles.use_progressive_refine,
        "max_bounces": cycles.max_bounces,
        # "min_bounces": cycles.min_bounces,
        "caustics_reflective": cycles.caustics_reflective,
        "caustics_refractive": cycles.caustics_refractive,
        "diffuse_bounces": cycles.diffuse_bounces,
        "glossy_bounces": cycles.glossy_bounces,
        "transmission_bounces": cycles.transmission_bounces,
        "volume_bounces": cycles.volume_bounces,
        # "transparent_min_bounces": cycles.transparent_min_bounces,
        "transparent_max_bounces": cycles.transparent_max_bounces,
        "blur_glossy": cycles.blur_glossy,
        "sample_clamp_indirect": cycles.sample_clamp_indirect


    }



    data = {
        "Terrain":terrain_data,
        "SunLight":light_data,
        "World":world_data,
        "Cycles":cycles_data
        
    }
    write_json(data, filepath)


def save_light_data(sun_az, sun_el, cam_name, store_dir):
    
    if not os.path.isdir(store_dir):
            os.makedirs(store_dir)
    filepath = store_dir + "/" + cam_name
    data = {
        "sun_az":sun_az,    # Sun azimuth angle (deg)
        "sun_el":sun_el     # Sun elevation angle (deg)
    }
    write_json(data, filepath + '_light.json')


def save_query_dataset_params(parser_args):
    
    if not os.path.isdir(parser_args.query_dest):
            os.makedirs(parser_args.query_dest)
    filepath = parser_args.query_dest + "/params.json"
    data = {
        "samples":parser_args.samples, # Number of query samples

        # Sun
        "sun_EL_range":parser_args.sun_EL_range, # Sun Elevation range (deg)
        "sun_AZ_range":parser_args.sun_AZ_range, # Sun Azimuth range (deg)
        "sun_EL_step":parser_args.sun_EL_step, # Sun Elevation step (deg)
        "sun_AZ_step":parser_args.sun_AZ_step, # Sun Azimuth step (deg)
        # Pose
        "altitude_range":parser_args.altitude_range, # Altitude range (m)
        "max_yaw":parser_args.max_yaw, # Max absolute value of yaw angle (deg)
        "max_pitch":parser_args.max_pitch, # Max absolute value of pitch angle (deg)
        "max_roll":parser_args.max_roll, # Max absolute value of roll angle (deg)

        # Path
        "query_path":parser_args.query_dest # Query dataset path
    }
    write_json(data, filepath)

def save_map_dataset_params(parser_args):
    
    if not os.path.isdir(parser_args.map_dest):
            os.makedirs(parser_args.map_dest)
    filepath = parser_args.map_dest + "/params.json"
    data = {
        # Sun
        "sun_EL_range":parser_args.sun_EL_range, # Sun Elevation range (deg)
        "sun_AZ_range":parser_args.sun_AZ_range, # Sun Azimuth range (deg)
        "sun_EL_step":parser_args.sun_EL_step, # Sun Elevation step (deg)
        "sun_AZ_step":parser_args.sun_AZ_step, # Sun Azimuth step (deg)
        # Path
        "map_path":parser_args.map_dest, # Map dataset path
        # Resolution:
        "map_px_res":parser_args.map_px_res, # m / pixels
        # Tiles
        "tiles_x":parser_args.tiles_x,
        "tiles_y":parser_args.tiles_y
    }
    write_json(data, filepath)


################## Data loading utilities ###########################################
    
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


def paste_map_tiles(tiles_dir, overall_width, overall_height, n_tiles_x, n_tiles_y):
    final_image_gray = Image.new('L', (overall_width, overall_height))
    
    img_id = 0
    cam_name = "OrthoCam"
    img_path = os.path.join(tiles_dir,cam_name + "_%04i" % img_id)

    # Paste each tile into the final image
    tile_width = overall_width // n_tiles_x
    tile_height = overall_height // n_tiles_y
    for y in range(n_tiles_y):
        for x in range(n_tiles_x):
            tile_path = img_path + f"_tile_{x}_{y}.png"
            tile_rgb = Image.open(tile_path) # load the tile RGB image
            tile_gray = tile_rgb.convert('L')   # Convert to gray
            final_image_gray.paste(tile_gray, (x * tile_width, (n_tiles_y - y - 1) * tile_height))

    # Convert the final image to a NumPy array
    final_image_gray_np = np.array(final_image_gray)

    return final_image_gray_np