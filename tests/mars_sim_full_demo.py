

#######################################
#######################################
# "Copyright 2023, by the California Institute of Technology. ALL RIGHTS RESERVED. 
# United States Government Sponsorship acknowledged. 
# Any commercial use must be negotiated with the Office of Technology 
# Transfer at the California Institute of Technology.
 
# This software may be subject to U.S. export control laws. 
# By accepting this software, the user agrees to comply with all applicable U.S. 
# export laws and regulations. User has the responsibility to obtain export licenses, 
# or other export authority as may be required before exporting such information to 
# foreign countries or providing access to foreign persons."
#######################################
#######################################


#######################################
## Graphical Utility for Moon surface simulation
## Authors: Dario Pisanti, Georgios Georgakis
#######################################


import os, signal
import sys
# Assuming this is ran in the root of the europa_sim repo. 
# Add current directory in path
BASE_PATH = os.getcwd() + "/"
sys.path.append(BASE_PATH)

import argparse
import bpy
import yaml
import numpy as np
import matplotlib.pyplot as plt
import utils
import imageio
import cv2
from utils import get_rot_tuple, get_dtm_metadata
from configs import Config
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"


def plot_correspondences(im1, im2, points1, points2, title1='From image 1', title2='To image 2'):
    color_str = "orange"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 12))
    ax1.imshow(im1, aspect='equal')
    ax1.set_title(title1)
    ax2.imshow(im2, aspect='equal')
    ax2.set_title(title2)
    for i in range(len(points1)):
        ax1.plot([points1[i,0]], [points1[i,1]], marker='*', markersize=15, color=color_str)
        ax1.text(points1[i,0]+5, points1[i,1]-10, str(i), fontsize=15, color=color_str)
        ax2.plot([points2[i,0]], [points2[i,1]], marker='*', markersize=15, color=color_str)
        ax2.text(points2[i,0]+5, points2[i,1]-10, str(i), fontsize=15, color=color_str)
    ax1.axis("off")
    ax2.axis("off")
    plt.tight_layout()
    plt.draw()
    # plt.show()

def show_image_pair(im1, im2, title1='Image 1', title2='Image 2'):
    color_str = "orange"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 12))
    ax1.imshow(im1, aspect='equal')
    ax1.set_title(title1)
    ax2.imshow(im2, aspect='equal')
    ax2.set_title(title2)
    ax1.axis("off")
    ax2.axis("off")
    plt.tight_layout()
    plt.draw()

def load_depth(path):
    # Using imageioi
    img = imageio.v2.imread(path, format='EXR-FI')
    dmap = img[..., 0]
    # # Using opencv
    # img = cv2.imread(f"./tmp/{scene.name}_depth0000.exr", cv2.IMREAD_UNCHANGED)
    # dmap = img[..., 2]
    dmap[dmap>10000] = 0 # Remove points that are far away
    # print(f"{scene.name}: dmap (min, max) = ({np.amin(dmap)},{np.max(dmap)})")
    return dmap

def normalize_depth(dmap, cam):
    if cam.data.type == "ORTHO":
        dmap[dmap <= cam.data.clip_start] = cam.location[2]
    dmap_norm = (dmap - np.amin(dmap))/np.abs(np.amax(dmap) - np.amin(dmap)) * 255.0 
    return dmap_norm

def validate_args(args):
    if (args.map_img_path is None) != (args.map_depth_path is None):
        parser.error("Both --map_img_path and --map_depth_path must be provided together, or neither of them.")

def backproject_ortho_points(points2D, depth, pixel_res, R, T):
    # TODO: account for non-square pixel (when aspect_x != aspect_y)
    height, width = np.shape(depth)
    points3D = np.zeros((points2D.shape[0],3)) #[]
    for i in range(len(points2D)):
        # Sample depth. TODO: Bilinear interpolation
        x = points2D[i][0] # need to verify x,y
        y = points2D[i][1]
        z = float(depth[int(y), int(x)])
        # z = bilinear_interpolation(depth, x, y)

        # local3D - Position of the point in the map camera frame in map camera coordinate
        local3D = np.zeros((3,1), dtype=np.float32)
        local3D[0] = (x - width/2) * pixel_res
        local3D[1] = (y - height/2) * pixel_res
        local3D[2] = z
        # print(f"Backproject from map - Local3D: {local3D}")

        w3D = np.dot(R,local3D) + T
        # print("w3D:", w3D.shape)
        # print("T:", T.shape)
    
        points3D[i,:] = w3D.transpose()
    return points3D

def backproject_persp_points(points2D, depth, intr, R, T):
    # print(f"\n***Inside backproject_persp_points****")
    # TODO: account for non-square pixel (when aspect_x != aspect_y)
    # TODO: acocunt for shift_x and shift_y of the persp camera
    fx, fy, cx, cy = intr[0], intr[1], intr[2], intr[3]
    points3D = np.zeros((points2D.shape[0], 3), dtype=np.float32)
    for i in range(len(points2D)):
        x = points2D[i][0] # need to verify x,y
        y = points2D[i][1]
        z = float(depth[int(y), int(x)])
        # z = bilinear_interpolation(depth, x, y)

        local3D = np.zeros((3,1), dtype=np.float32)
        local3D[0] = (x-cx)*z / fx
        local3D[1] = (y-cy)*z / fy
        local3D[2] = z

        w3D =  np.dot(R,local3D) + T # Column vector
        # print(f"\nx: {x}, y:{y}, z:{z}")
        # print(f"local3D: {local3D}")
        # print(f"R: {R}")
        # print(f"T: {T}")
        # print(f"wl3D: {w3D}")
       
        points3D[i,:] = w3D.transpose()
    return points3D

def project_on_persp(points3D, intr, R, T):
    # print(f"\n***Inside project_on_persp****")
    # TODO: account for non-square pixel (when aspect_x != aspect_y)
    # TODO: acocunt for shift_x and shift_y of the persp camera
    fx, fy, cx, cy = intr[0], intr[1], intr[2], intr[3]
    points2D = np.zeros((len(points3D), 2), dtype=float)
    # print(f"points2D: {points2D}")
    point3D = np.zeros((3,1), dtype=np.float32)
    for i in range(len(points3D)):
        point3D[:,0] = points3D[i,:].transpose()
        local3D = np.dot(R,point3D) + T # Column vector
        # print(f"\nPoint3D: {point3D}")
        # print(f"R: {R}")
        # print(f"T: {T}")
        # print(f"local3D: {local3D}")
        X, Y, Z = local3D[0,0], local3D[1,0], local3D[2,0]
        # Project local 3D points from camera frame to image frame
        # print(f"X: {X}, Y: {Y}, Z: {Z}")
        # print(f"cx: {cx}, cy: {cy}")
        # print(f"fx: {fx}, fy: {fy}")
        u = cx + fx*X/Z
        v = cy + fy*Y/Z

        points2D[i,0] = u
        points2D[i,1] = v
        # print(f"x:{u}, y:{v}")
    return points2D

def project_on_ortho(points3D, pixel_res, cx, cy, R, T):
    # Project 3D world points on map plane
    # points3D Nx3
    # pixe_res_x (y): pixel resolution [m/pixel] along x (y)
    # TODO: account for non-square pixel (when aspect_x != aspect_y)
    points2D = np.zeros((len(points3D), 2), dtype=float)
    point3D = np.zeros((3,1), dtype=np.float32)
    for i in range(len(points3D)):
        point3D[:,0] = points3D[i,:].transpose()
        local3D = np.dot(R, point3D) + T
        # Project local 3D points from camera frame to image frame
        X, Y, Z = local3D[0,0] , local3D[1,0], local3D[2,0]
        u =  X*(1/pixel_res) + cx
        v =  Y*(1/pixel_res) + cy

        points2D[i,0] = u
        points2D[i,1] = v
    return points2D

def get_cam2cam_correspondences(cam0_scene, cam0, cam0_gray, cam0_depth, cam1_scene, cam1, cam1_gray, cam1_depth, bound=0, px_sampling_step = 1000, plot_on_dmap=False):
   
    # Load first perspective cam data
    #   Cam paremeters
    cam0_img_width, cam0_img_height = cam0_scene.render.resolution_x, cam0_scene.render.resolution_y
    cam0_img_aspect_x, cam0_img_aspect_y = cam0_scene.render.pixel_aspect_x, cam0_scene.render.pixel_aspect_y
    cam0_img_aspect_ratio = cam0_img_aspect_x / cam0_img_aspect_y
    cam0_cx, cam0_cy = cam0_img_width / 2., cam0_img_height / 2.
    cam0_clip_start = cam0.data.clip_start
    if cam0.data.type == 'PERSP':
        cam0_fx = (cam0_img_width / cam0.data.sensor_width) * cam0.data.lens
        cam0_fy = cam0_img_aspect_ratio * cam0_fx
        cam0_intr = (cam0_fx, cam0_fy, cam0_cx, cam0_cy)
    elif cam0.data.type == 'ORTHO':
        cam0_ortho_scale_x = cam0.data.ortho_scale
        cam0_px_res = cam0_ortho_scale_x / cam0_img_width
    else:
        raise(f"Camera projection model not recognized.")
    #   Cam poses
    R_cam0_wc = np.matmul(utils.Eul312toSO3(cam0.rotation_euler.x, cam0.rotation_euler.y, cam0.rotation_euler.z), utils.XtoSO3(np.pi)) # the convention in wc is the other way round
    R_cam0_cw = R_cam0_wc.T
    t_cam0_wc_w = np.transpose(np.array(cam0.location)) # Column vector
    t_cam0_wc_w = t_cam0_wc_w.reshape((3, 1))
    t_cam0_cw_w = -t_cam0_wc_w
    t_cam0_cw_c = np.dot(R_cam0_cw, t_cam0_cw_w)


    # Load second perspective cam data
    #   Cam paremeters
    cam1_img_width, cam1_img_height = cam1_scene.render.resolution_x, cam1_scene.render.resolution_y
    cam1_img_aspect_x, cam1_img_aspect_y = cam1_scene.render.pixel_aspect_x, cam1_scene.render.pixel_aspect_y
    cam1_img_aspect_ratio = cam1_img_aspect_x / cam1_img_aspect_y
    cam1_clip_start = cam1.data.clip_start
    cam1_cx, cam1_cy = cam1_img_width / 2., cam1_img_height / 2.
    if cam1.data.type == 'PERSP':
        cam1_fx = (cam1_img_width / cam1.data.sensor_width) * cam1.data.lens
        cam1_fy = cam1_img_aspect_ratio * cam1_fx
        cam1_intr = (cam1_fx, cam1_fy, cam1_cx, cam1_cy)
    elif cam1.data.type == 'ORTHO':
        cam1_ortho_scale_x = cam1.data.ortho_scale
        cam1_px_res = cam1_ortho_scale_x / cam1_img_width
    else:
        raise(f"Camera projection model not recognized.")
    #   Cam poses
    R_cam1_wc = np.matmul(utils.Eul312toSO3(cam1.rotation_euler.x, cam1.rotation_euler.y, cam1.rotation_euler.z), utils.XtoSO3(np.pi)) # the convention in wc is the other way round
    R_cam1_cw = R_cam1_wc.T
    t_cam1_wc_w = np.transpose(np.array(cam1.location)) # Column vector
    t_cam1_wc_w = t_cam1_wc_w.reshape((3, 1))
    t_cam1_cw_w = -t_cam1_wc_w
    t_cam1_cw_c = np.dot(R_cam1_cw, t_cam1_cw_w)

    # Filter indices to account for the boundary and valid depth
    inds = np.where(cam0_depth > cam0_clip_start)
    # Filter indices to account for the boundary
    valid_inds = (inds[0] >= bound) & (inds[0] < cam0_depth.shape[0] - bound) & \
                (inds[1] >= bound) & (inds[1] < cam0_depth.shape[1] - bound)
    inds = (inds[0][valid_inds], inds[1][valid_inds])
    # print(inds)
    cam0_points_2D = np.zeros((len(inds[0]), 2), dtype=float)
    cam0_points_2D[:,0] = inds[1] # inds[1] is x (width coordinate)
    cam0_points_2D[:,1] = inds[0]
    cam0_points_2D = cam0_points_2D[::px_sampling_step, :] # Subsample points

    # Unproject cam0 2D points into camera coordinate frame...
    if cam0.data.type == 'PERSP':
        cam0_points_3D = backproject_persp_points(cam0_points_2D, cam0_depth, cam0_intr, R=R_cam0_wc, T=t_cam0_wc_w) # N x 3
    elif cam0.data.type == 'ORTHO':
        cam0_points_3D = backproject_ortho_points(cam0_points_2D, cam0_depth, cam0_px_res, R=R_cam0_wc, T=t_cam0_wc_w) # N x 3
    else:
        raise(f"Camera projection model not recognized.")
    
    # ... and project on cam1
    if cam1.data.type == 'PERSP':
        cam1_points_2D = project_on_persp(cam0_points_3D, cam1_intr, R=R_cam1_cw, T=t_cam1_cw_c)
    elif cam1.data.type == 'ORTHO':
        cam1_points_2D = project_on_ortho(cam0_points_3D, cam1_px_res, cx=cam1_cx, cy=cam1_cy, R=R_cam1_cw, T=t_cam1_cw_c)
    else:
        raise(f"Camera projection model not recognized.")
    

    # Find points within the image bounds
    valid_cam0_points_2D = []
    valid_cam0_points_3D = []
    valid_cam0_points_2D_reproj =[]
    valid_cam1_points_2D = []
    valid_cam1_points_3D = []

    for i in range(cam1_points_2D.shape[0]):
        x,y = cam1_points_2D[i,:]
        if x>bound and y>bound and x<cam1_img_width-bound-1 and y<cam1_img_height-bound-1:
            if cam1.data.type == 'PERSP':
                cam1_point_3D = backproject_persp_points(np.expand_dims(cam1_points_2D[i,:], axis=0), cam1_depth, intr=cam1_intr, R=R_cam1_wc, T=t_cam1_wc_w)
            elif cam1.data.type == 'ORTHO':
                cam1_point_3D = backproject_ortho_points(np.expand_dims(cam1_points_2D[i,:], axis=0), cam1_depth, cam1_px_res, R=R_cam1_wc, T=t_cam1_wc_w)
            else:
                raise(f"Camera projection model not recognized.")
            # Check if the point is co-visible from persp camera
            point_3D_err = np.linalg.norm(cam0_points_3D[i,:]- cam1_point_3D, axis=1) # m
            if point_3D_err < 0.1:
                if cam0.data.type == 'PERSP':
                    cam0_point_2D_reproj = project_on_persp(cam1_point_3D, cam0_intr, R=R_cam0_cw, T=t_cam0_cw_c)
                elif cam0.data.type == 'ORTHO':
                    cam0_point_2D_reproj = project_on_ortho(cam1_point_3D, cam0_px_res, cx=cam0_cx, cy=cam0_cy, R=R_cam0_cw, T=t_cam0_cw_c)
                else:
                    raise(f"Camera projection model not recognized.")
                valid_cam0_points_2D.append(cam0_points_2D[i,:])
                valid_cam0_points_3D.append(cam0_points_3D[i,:])
                valid_cam0_points_2D_reproj.append(cam0_point_2D_reproj[0])

                valid_cam1_points_2D.append(cam1_points_2D[i,:])
                valid_cam1_points_3D.append(cam1_point_3D[0])


    
    cam0_points_2D = np.asarray(valid_cam0_points_2D)
    cam0_points_2D_reproj = np.asarray(valid_cam0_points_2D_reproj)
    cam0_points_3D = np.asarray(valid_cam0_points_3D)

    cam1_points_2D = np.asarray(valid_cam1_points_2D)
    cam1_points_3D = np.asarray(valid_cam1_points_3D)
    
    # Further subsampling
    # cam0_points_2D = cam0_points_2D[::50,:]
    # cam1_points_2D_proj = cam1_points_2D_proj[::50,:]
    # cam0_points_3D = cam0_points_3D[::50,:]

    # Plot
    plot_correspondences(im1=cam0_gray, 
                         im2=cam1_gray, 
                        points1=cam0_points_2D, 
                        points2=cam1_points_2D,
                        title1 = f"From {cam0.name}", title2 = f"To {cam1.name}")

    plot_correspondences(im1=cam1_gray, 
                         im2=cam0_gray, 
                        points1=cam1_points_2D,
                        points2=cam0_points_2D_reproj,
                        title1=f"From {cam1.name}", title2=f"To {cam0.name}")
    
    if plot_on_dmap:
        cam0_depth_img = normalize_depth(cam0_depth, cam0)
        cam1_depth_img = normalize_depth(cam1_depth, cam1)
        plot_correspondences(im1=cam0_depth_img, 
                        im2=cam1_depth_img, 
                        points1=cam0_points_2D, 
                        points2=cam1_points_2D,
                        title1 = "From (Depth) {cam1.name}", title2 = f"To (Depth) {cam1.name}")
     

    print(f"\n**** {cam0.name} points reprojection error ({cam0.name} -> {cam1.name} -> {cam0.name}) **********")
    repr_err = np.linalg.norm(cam0_points_2D_reproj- cam0_points_2D, axis=1)
    print(f"shape: {repr_err.shape}")
    k = 40
    top_k_ids = np.argsort(repr_err)[-k:][::-1]
    top_k = repr_err[top_k_ids]
    print(f"Max: {np.amax(repr_err)}, id: {np.argmax(repr_err)}, 3D point from {cam0.name}: {cam0_points_3D[np.argmax(repr_err),:]}, from {cam1.name}: {cam1_points_3D[np.argmax(repr_err),:]}")
    print(f"Min: {np.amin(repr_err)}, id: {np.argmin(repr_err)}, 3D point from {cam0.name}: {cam0_points_3D[np.argmin(repr_err),:]}, from {cam1.name}: {cam1_points_3D[np.argmin(repr_err),:]}")
    print(f"Mean: {np.mean(repr_err)}")
    print(f"Std: {np.std(repr_err)}")
    print(f"Top {k} Max: {top_k}, ids: {top_k_ids}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=".")

    parser.add_argument('--main_yaml', type=str, default='moon.yaml')
    parser.add_argument('--keep_blender_running', default=False, action='store_true', help="If enabled, kills blender process after rendering." )
    parser.add_argument('--sun_az', type=float, default=0., help='Sun Azimuth angle (deg)')
    parser.add_argument('--sun_el', type=float, default=90., help='Sun Elevation angle (deg)')
    parser.add_argument('--map_img_path', type=str, default=None, help='Path to an already rendered map image. Default:None. If provided, the demo will use that without rendering a new map.')
    parser.add_argument('--map_depth_path', type=str, default=None, help='Path to an already rendered depth image. If provided, the demo will use that without rendering a new map.')
    parser.add_argument('--bound', type=float, default=0., help='Defines the margin (in pixels) to exclude from the edges of the images during points projections.')

    cam0_parser = parser.add_argument_group('Camera 0', 'Group of arguments for camera 0')
    cam0_parser.add_argument('--cam0_altitude', type=float, default=700., help='Altitude of the persp camera (m)')
    cam0_parser.add_argument('--cam0_loc_x', type=float, default=-1580., help='X Location of the persp cam in Blender reference frame (m)')
    cam0_parser.add_argument('--cam0_loc_y', type=float, default=2510., help='Y Location of the persp cam in Blender reference frame (m)')
    cam0_parser.add_argument('--cam0_yaw', type=float, default=0., help='Yaw (deg) - Angle around Z axis (pointing upwards)')
    cam0_parser.add_argument('--cam0_pitch', type=float, default=0., help='Pitch (deg) - Angle around X axis (pointing rightward w.r.t. the direction of camera motion)')
    cam0_parser.add_argument('--cam0_roll', type=float, default=0., help='Roll (deg) - Angle around Y axis (pointing towards the direction of camera motion)')
    cam0_parser.set_defaults()

    cam1_parser = parser.add_argument_group('Camera 1', 'Group of arguments for camera 1')
    cam1_parser.add_argument('--cam1_altitude', type=float, default=700., help='Altitude of the persp camera (m)')
    cam1_parser.add_argument('--cam1_loc_x', type=float, default=-1480., help='X Location of the persp cam in Blender reference frame (m) - for negative values be sure to provide the value in quote and with space: e.g. " -1678"')
    cam1_parser.add_argument('--cam1_loc_y', type=float, default=2710., help='Y Location of the persp cam in Blender reference frame (m)')
    cam1_parser.add_argument('--cam1_yaw', type=float, default=0., help='Yaw (deg) - Angle around Z axis (pointing downwards)')
    cam1_parser.add_argument('--cam1_pitch', type=float, default=0., help='Pitch (deg) - Angle around Y axis (pointing rightward w.r.t. the direction of motion)')
    cam1_parser.add_argument('--cam1_roll', type=float, default=0., help='Roll (deg) - Angle around X axis (pointing towards the direction of motion)')
    cam1_parser.set_defaults()

    print("sys.argv:", sys.argv)

    if '--' not in sys.argv:
        argv = []
    else:
        argv = sys.argv[sys.argv.index('--') + 1:]

    print("argv passed to parser:", argv)
    args = parser.parse_args(argv)
    print("Parsed arguments:", args)

    validate_args(args) # Check that both map_img_path and map_depth path arguments are provided whenever one of them is provided

    # Import add-on
    addon_name = "addon_ground_truth_generation"
    addon_path = f"./blender_addons/VisionBlender/{addon_name}.py"
    utils.install_addon(addon_name, addon_path)
        
    # Install HiRISE DTM importer
    addon_name = "dtmimporter"
    addon_path = f"./blender_addons/hirise_dtmimporter/{addon_name}.zip"
    utils.install_addon(addon_name, addon_path)


    # LOAD ENVIRONMENT CONFIGURATION
    # Load the yaml with the enviornment configuration
    env_config_manager = Config(BASE_PATH + 'configs/'+args.main_yaml)
    env_config = env_config_manager.get_config()
    # Load th blend file (doesn't work from "blender start")
    file_name = env_config['Terrain']['blend_file']
    blend_path=f"{file_name}.blend"
    utils.load_blend_file(blend_path)

    # # Remove all the objects except the terrain
    # print(bpy.data.objects)
    # for obj in bpy.data.objects:
    #     print("name:", obj.name)
    #     if (obj.name=="Cube") or (obj.name=="Lamp") or (obj.name=="Light") or (obj.name=="Camera"):
    #         obj.select_set(True)
    #     elif obj.name=="Terrain":
    #         obj.select_set(False)
    # bpy.ops.object.delete()

    # Import Terrain with metadata
    terrain_mesh = bpy.data.objects[env_config['Terrain']["terrain_name"]]
    dtm_metadata = get_dtm_metadata(terrain_mesh, print_data=True)
    map_ortho_width = dtm_metadata['map_size'][1]
    map_ortho_height = dtm_metadata['map_size'][0]  # m
    if args.map_img_path:
        map_depth = load_depth(args.map_depth_path)
        map_gray = cv2.imread(args.map_img_path)
        # print(map_gray.shape)
        map_px_res = map_ortho_width / map_gray.shape[1]
    else:
        map_px_res = 10 # map pixel resolution [m /pixel] (the texture resolution for jezer.yaml is 1 m/pixel)


    # LOAD AND SAVE CAMERA CONFIGURATIONS FOR THE SPECIFIC TERRAIN
    # Load the yaml with the defualt camera configuration
    cam0_config_manager = Config(BASE_PATH + 'configs/base_cam/persp_cam.yaml')
    cam1_config_manager = Config(BASE_PATH + 'configs/base_cam/persp_cam.yaml')
    mapcam_config_manager = Config(BASE_PATH + 'configs/base_cam/ortho_cam.yaml')
    # Save the new config file for the specified terrain
    
    mapcam_config_manager.update_config({
        'ortho_scale':map_ortho_width,
        'img_width':int(map_ortho_width/map_px_res),
        'img_height':int(map_ortho_height/map_px_res)
    })
    cam0_config = cam0_config_manager.get_config()
    cam1_config = cam1_config_manager.get_config()
    mapcam_config = mapcam_config_manager.get_config()
    # Optional: save new configuration files
    cam0_config_manager.save_config(BASE_PATH + 'configs/jezero_persp_cam0.yaml')
    cam1_config_manager.save_config(BASE_PATH + 'configs/jezero_persp_cam1.yaml')
    mapcam_config_manager.save_config(BASE_PATH + 'configs/jezero_ortho_cam.yaml')
    

    # Setting rendering engine and light
    utils.set_cycles(samples=env_config['cycles_samples'])
    light = utils.add_light_source(light_type="SUN", config=env_config['SunLight'], name="SunLight")

    # Generate two linked scenes for map and queries rendering
    cam0_scene_name = "Persp0"
    cam1_scene_name = "Persp1"
    map_scene_name = "Ortho"
    bpy.context.scene.name = cam0_scene_name
    bpy.ops.scene.new(type='LINK_COPY')
    bpy.context.scene.name = cam1_scene_name
    bpy.ops.scene.new(type='LINK_COPY')
    bpy.context.scene.name = map_scene_name
    cam0_scene = bpy.data.scenes[cam0_scene_name]
    cam1_scene = bpy.data.scenes[cam1_scene_name]
    map_scene = bpy.data.scenes[map_scene_name]

    # Set Cam0 scene
    cam0_name = "PerspCam0"
    utils.set_scene(scene=cam0_scene, cam_config=cam0_config, world_config=env_config)
    cam0 = utils.add_camera(cam0_config, name=cam0_name) # Manually add the cameras
    cam0_scene.collection.objects.link(cam0) # Link to the scene

    # Set Cam1 scene
    cam1_name = "PerspCam1"
    utils.set_scene(scene=cam1_scene, cam_config=cam1_config, world_config=env_config)
    cam1 = utils.add_camera(cam1_config, name=cam1_name) # Manually add the cameras
    cam1_scene.collection.objects.link(cam1) # Link to the scene
    print(f"\nAdded Objects:", bpy.data.objects.keys())

    # Set Map scene
    map_cam_name = "OrthoCam"
    utils.set_scene(scene=map_scene, cam_config=mapcam_config, world_config=env_config)
    map_cam = utils.add_camera(mapcam_config, name=map_cam_name) # Manually add the cameras
    map_scene.collection.objects.link(map_cam) # Link to the scene

    print(f"\nAdded Objects:", bpy.data.objects.keys())


    # Set Sun position
    light.rotation_euler = get_rot_tuple((0, 90. - args.sun_el, args.sun_az))
  
    # Set persp  pose
    #   The camera object frame (O) is initially aligned with the Blender world reference frame and defined as follows:
    #   Z axis (pointing upwards)
    #   Y axis (pointing towards the direction of  camera motion)
    #   X axis (pointing rightward)
    #   The camera boresight axis is pointing along -Z
    #   The Euler sequence is: 3 1 2 (rot_Z, rot_X, rot_Y)
    #   The corresponding Blender Euler mode is: YXZ
    #   With rotation angles [0, 0, 0] the camera boresight axis is pointing downward
    #   rot_Z = Yaw
    #   rot_X = Pitch
    #   rot Y = Roll
    #   The camera image frame (C) has origin in the camera object frame (O):
    #   it is defined by applying a rotation of 180 deg to C frame, around its X axis
    
    # Set Cam 0 pose
    utils.set_camera_above_terrain(cam0, terrain_mesh, args.cam0_loc_x, args.cam0_loc_y, args.cam0_altitude)  
    cam0_loc = cam0.location
    cam0_rot = [args.cam0_pitch, args.cam0_roll, args.cam0_yaw] # X, Y, Z euler angles in B frame (deg) ([0,0,0] points downwards)
    cam0.rotation_euler = get_rot_tuple(cam0_rot)

    # Set Cam 1 pose
    utils.set_camera_above_terrain(cam1, terrain_mesh, args.cam1_loc_x, args.cam1_loc_y, args.cam1_altitude)  
    cam1_loc = cam1.location
    cam1_rot = [args.cam1_pitch, args.cam1_roll, args.cam1_yaw] # X, Y, Z euler angles in B frame (deg) ([0,0,0] points downwards)
    cam1.rotation_euler = get_rot_tuple(cam1_rot)

    # Set Ortho cam pose
    utils.set_camera_above_terrain(map_cam, terrain_mesh, env_config["ortho_cam_x"], env_config["ortho_cam_y"], env_config["ortho_cam_altitude"])  
    map_cam.rotation_euler = (0,0,0)


    # Renders
    utils.render(cam0_scene, cam0, f"tmp/{cam0_scene.name}")
    utils.render(cam1_scene, cam1, f"tmp/{cam1_scene.name}")
    # Get images and depth maps
    #   Depth
    cam0_depth = load_depth(f"./tmp/{cam0_scene.name}_depth0001.exr")
    cam1_depth = load_depth(f"./tmp/{cam1_scene.name}_depth0001.exr")
    #   Map image
    cam0_gray = cv2.imread(f"tmp/{cam0_scene.name}.png")
    cam1_gray = cv2.imread(f"tmp/{cam1_scene.name}.png")    
    
    if args.map_img_path is None:
        utils.render(map_scene, map_cam, f"tmp/{map_scene.name}")
        map_depth = load_depth(f"./tmp/{map_scene.name}_depth0001.exr")
        map_gray = cv2.imread(f"tmp/{map_scene.name}.png")

    ##### Print data
    # Poses
    print(f"--- Cam 0 pose---")
    print(f"Location in Blender world reference frame: (x, y, z) = ({cam0_loc[0]} , {cam0_loc[1]}, {cam0_loc[2]})")
    print(f"Altitude: {args.cam0_altitude} m") 
    print(f"Euler angles: (Yaw, Pitch, Roll) = {(args.cam0_yaw, args.cam0_pitch, args.cam0_roll)} deg")

    print(f"--- Cam 1 pose---")
    print(f"Location in Blender world reference frame: (x, y, z) = ({cam1_loc[0]} , {cam1_loc[1]}, {cam1_loc[2]})")
    print(f"Altitude: {args.cam1_altitude} m") 
    print(f"Euler angles: (Yaw, Pitch, Roll) = {(args.cam1_yaw, args.cam1_pitch, args.cam1_roll)} deg")

    print(f"------------ Map camera pose ------------")
    print(f"Location in Blender world reference frame: (x, y, z) = ({map_cam.location[0]} , {map_cam.location[1]} , {map_cam.location[2]})")
    print(f"Euler angles: (Yaw, Pitch, Roll) = (0, 0, 0) deg")
    print(f"------------ Save Metadata ------------")

    print(f"\n------------ Sun Angle ------------")
    print(f"(Azimuth, Elevation) = ({args.sun_az} , {args.sun_el} ) deg")


    ##### Project points from Map to Cam0 and viceversa
    get_cam2cam_correspondences(map_scene, map_cam, map_gray, map_depth, cam0_scene, cam0, cam0_gray, cam0_depth, bound=0, px_sampling_step=100, plot_on_dmap=True)
    get_cam2cam_correspondences(cam0_scene, cam0, cam0_gray, cam0_depth, cam1_scene, cam1, cam1_gray, cam1_depth, bound=0, px_sampling_step=100, plot_on_dmap=True)
    # get_cam2cam_correspondences(cam0_scene, cam0. cam0_gray, cam0_depth, map_scene, map_cam, map1_gray, map1_depth, bound=0, px_sampling_step=1, plot_on_dmap=False)
    

    ##### Show depth maps
    # map_depth_img = normalize_depth(load_depth(map_scene), map_cam)
    # cam0_depth_img = normalize_depth(load_depth(cam0_scene), cam0)
    # show_image_pair(map_depth_img, cam0_depth_img, title1='Map', title2='Cam0')
    # cam0_gray = cv2.imread(f"tmp/{cam0_scene.name}.png")
    # show_image_pair(cam0_gray, cam0_depth_img, title1='Cam0 Image', title2='Cam0 Depth')

    # Show map:
    map_gray = cv2.imread(map_gray)
    fig, ax = plt.subplots()
    ax.imshow(map_gray, aspect='equal')
    ax.set_title("Map")
    ax.axis("off")
    plt.tight_layout()

    plt.show(block=True)

    if not args.keep_blender_running:
        # quit when done
        bpy.ops.wm.quit_blender() 

        # Manually kill blender process in case the cmd above does not work
        # This is needed when we are generating a dataset using the run_gen_datasets.sh (product of gen_yamls.py)
        # If blender is not killed, then the next command in the shell script does not initiate
        
        # iterating through each instance of the process
        for line in os.popen("ps ax | grep blender | grep -v grep"):
            fields = line.split()
                
            # extracting Process ID from the output
            pid = fields[0]
                
            # terminating process
            os.kill(int(pid), signal.SIGKILL)

    print("Finished!")    

    
