import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import random
import utils

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

def plot_points(im, points2D):
    color_str = 'orange'
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 12))
    ax1.imshow(im, aspect='equal')
    for i in range(len(points2D)):
        ax1.plot([points2D[i,0]], [points2D[i,1]], marker='*', markersize=15, color=color_str)
        ax1.text(points2D[i,0]+5, points2D[i,1]-10, str(i), fontsize=15, color=color_str)
    ax1.axis("off")
    plt.tight_layout()
    plt.draw()
    # plt.show()

def backproject_map_points(points2D, depth, pixel_resolution, R, T):
    
    height, width = np.shape(depth)
    points3D = np.zeros((points2D.shape[0],3)) #[]
    for i in range(len(points2D)):
        # Sample depth. TODO: Bilinear interpolation
        x = points2D[i][0] # need to verify x,y
        y = points2D[i][1]
        z = depth[int(y), int(x)]

        # local3D - Position of the point in the map camera frame in map camera coordinate
        local3D = np.zeros((3,1), dtype=np.float32)
        local3D[0] = (x - width/2) * pixel_resolution
        local3D[1] = (y - height/2) * pixel_resolution
        local3D[2] = z

        w3D = np.dot(R.transpose(), local3D - T)  # Column vector
        #print("w3D:", w3D)
    
        points3D[i,:] = w3D.transpose()
    return points3D

def backproject_query_points(points2D, depth, intr, R, T):

    fx, fy, cx, cy = intr[0], intr[1], intr[2], intr[3]
    points3D = np.zeros((points2D.shape[0], 3), dtype=np.float32)
    for i in range(len(points2D)):
        x = points2D[i][0] # need to verify x,y
        y = points2D[i][1]
        z = depth[int(y), int(x)]

        local3D = np.zeros((3,1), dtype=np.float32)
        local3D[0] = (x-cx)*z / fx
        local3D[1] = (y-cy)*z / fy
        local3D[2] = z
        w3D = np.dot(R.transpose(), local3D - T) # Column vector
        #print("w3D:", w3D)
        
        points3D[i,:] = w3D.transpose()
    return points3D

def project_on_query(points3D, intr, R, T):
    
    fx, fy, cx, cy = intr[0], intr[1], intr[2], intr[3]
    points2D = np.zeros((len(points3D), 2), dtype=float)
    for i in range(len(points3D)):
        local3D = np.dot(R,points3D[i,:]) + np.transpose(T) # Column vector
        X, Y, Z = local3D[0,0], local3D[0,1], local3D[0,2]
        # Project local 3D points from camera frame to image frame
        u = cx + fx*X/Z
        v = cy + fy*Y/Z

        points2D[i,0] = u
        points2D[i,1] = v
    return points2D

def project_on_map(points3D, pixel_res, cx, cy, R, T):
    # Project 3D world points on map plane
    # points3D Nx3
    # pixe_res: pixel resolution [m/pixel] along
    points2D = np.zeros((len(points3D), 2), dtype=float)
    for i in range(len(points3D)):
        local3D = np.dot(R, points3D[i,:]) + np.transpose(T)
        # Project local 3D points from camera frame to image frame
        X, Y, Z = local3D[0,0], local3D[0,1], local3D[0,2]
        u =  X*(1/pixel_res) + cx
        v =  Y*(1/pixel_res) + cy

        points2D[i,0] = u
        points2D[i,1] = v
    return points2D



# Project points from the map to corresponding images and viceversa
# Will need this functionality when generating ground-truth for training a learned descriptor

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument('--map_path', type=str, default='./visualizations/sample_render/maps/', required=False)
    parser.add_argument("--query_path", default='./visualizations/sample_render/queries/pitch', help="Dataset with query images", required=False)
    parser.add_argument('--map_id', type=str, default='MapCam0000', required=False)
    parser.add_argument("--query_id", default='QueryCam0001', required=False)
    args = parser.parse_args()

    ############################################################################
    # Notation:
    # d: drone reference frame 
    # w: world reference frame (Blender frame)
    # b: camera body frame (initially aligned to Blender frame) 
    # c: camera frame
    # R_wc: Rotation matrix from w-frame to c-frame
    # t_cw_w: Vector from c-frame orgin to w-frame origin in w coordinates

    ############################################################################

    ############# LOAD MAP DATA ###############
    # Load map camera parameters metadata
    with open(os.path.join(args.map_path, f"cam_data/{args.map_id}_cam.json")) as f:
        map_cam_data = json.load(f)
    map_ortho_width = map_cam_data['ortho_width']       # m
    map_ortho_height = map_cam_data['ortho_height']     # m
    map_img_width = map_cam_data['img_width']         # pixel
    map_img_height = map_cam_data['img_height']       # pixel
    map_clip_start = map_cam_data['clip_start']
    map_cx, map_cy = map_img_width / 2, map_img_height / 2
    # Load map pose metadata
    with open(os.path.join(args.map_path, f"cam_pose/{args.map_id}_pose.json")) as f:
        map_pose = json.load(f)
    map_yaw, map_pitch, map_roll = map_pose["drone_ypr"] # Euler angles in drone body frame (rad)
    map_rot_z, map_rot_x, map_rot_y = -map_yaw, map_pitch, map_roll    # Euler angles in camera blender object frame (rad)
    R_map_wc = utils.X2SO3(np.pi)*utils.Eul312toSO3(map_rot_x, map_rot_y, map_rot_z) # the convention in wc is the other way round
    R_map_cw = R_map_wc.transpose()
    t_map_wc_w = np.transpose(np.array([map_pose['t_wc_w']])) # Column vector
    t_map_cw_w = -t_map_wc_w
    t_map_cw_c = R_map_cw * t_map_cw_w
    # Load map image and depth
    map_depth = np.load(os.path.join(args.map_path, f"depth/{args.map_id}_depth.npy")) + map_clip_start
    map_gray = cv2.imread(os.path.join(args.map_path, f"images/{args.map_id}.png"))


    ############# LOAD QUERY DATA ###############
    # Load query camera parameters metadata
    with open(os.path.join(args.query_path, f"cam_data/{args.query_id}_cam.json")) as f:
        query_cam_data = json.load(f)
    query_img_width = query_cam_data['img_width']
    query_img_height = query_cam_data['img_height']
    query_fx = query_cam_data['fx'] # pixel
    query_fy = query_cam_data['fy'] # pixel
    query_clip_start = query_cam_data['clip_start']
    query_cx, query_cy = query_cam_data['cx'], query_cam_data['cy']
    # Load query pose metadata
    with open(os.path.join(args.query_path, f"cam_pose/{args.query_id}_pose.json")) as f:
        query_pose = json.load(f)
    query_yaw, query_pitch, query_roll = query_pose["drone_ypr"] # Euler angles in drone body frame (rad)
    query_rot_z, query_rot_x, query_rot_y = -query_yaw, query_pitch, query_roll    # Euler angles in camera blender object frame (rad)
    R_query_wc = utils.X2SO3(np.pi)*utils.Eul312toSO3(query_rot_x, query_rot_y, query_rot_z)
    R_query_cw = R_query_wc.transpose()
    t_query_wc_w = np.transpose(np.array([query_pose['t_wc_w']])) # Colun vector
    t_query_cw_w = -t_query_wc_w
    t_query_cw_c = R_query_cw * t_query_cw_w
    # Load query image and depth
    query_depth = np.load(os.path.join(args.query_path, f"depth/{args.query_id}_depth.npy")) + query_clip_start
    query_gray = cv2.imread(os.path.join(args.query_path, f"images/{args.query_id}.png"))





    ########### Project 2D map points on query image #############################
    inds = np.where(map_depth > map_clip_start)
    map_points_2D = np.zeros((len(inds[0]), 2), dtype=float)
    map_points_2D[:,0] = inds[1] # inds[1] is x (width coordinate)
    map_points_2D[:,1] = inds[0]

    # Subsample map_points
    map_points_2D = map_points_2D[::1000, :]
    #print("Map points:", map_points_2D)
    #print(map_points_2D.shape)

    # Unproject map 2D points into map camera coordinate frame
    px_res = map_ortho_width/map_img_width
    map_points_3D = backproject_map_points(map_points_2D, map_depth, px_res, R=R_map_cw, T=t_map_cw_c) # N x 3
    #print(map_point_3D)
    #print(map_point_3D.shape)
    intr = (query_fx, query_fy, query_cx, query_cy)
    query_points_2D = project_on_query(map_points_3D, intr, R=R_query_cw, T=t_query_cw_c)

    # Find points within the image bounds
    valid_query_points = []
    #valid_inds = []
    valid_map_points = []
    for i in range(query_points_2D.shape[0]):
        x,y = query_points_2D[i,:]
        
        if x>0 and y>0 and x<query_img_width-1 and y<query_img_height-1:
            valid_query_points.append(query_points_2D[i,:])
            #valid_inds.append(i)
            valid_map_points.append(map_points_2D[i,:])

    valid_query_points = np.asarray(valid_query_points)
    #valid_inds = np.asarray(valid_inds)
    valid_map_points = np.asarray(valid_map_points)
    #print(valid_query_points.shape)
    #print(valid_map_points.shape)

    plot_correspondences(im1=map_gray, 
                        im2=query_gray, 
                        points1=valid_map_points[::10,:], 
                        points2=valid_query_points[::10,:],
                        title1 = "From Map", title2 = "To Query")

    print(f"From map: \n{valid_map_points[::10,:]}")
    print(f"to query: \n{valid_query_points[::10,:]}")
    


    ########### Project 2D query points on map image #############################
    # query_points_2D = np.asarray([[query_cx, query_cy], [0,0], [query_width-1,0], [query_width-1, query_height-1], [0,query_height-1]])
    query_points_2D = valid_query_points[::10,:]
    query_points_3D = backproject_query_points(query_points_2D, query_depth, intr=(query_fx, query_fy, query_cx, query_cy), R=R_query_cw, T=t_query_cw_c)
    map_points_2D = project_on_map(query_points_3D, px_res, cx=map_cx, cy=map_cy, R=R_map_cw, T=t_map_cw_c)

    plot_correspondences(im1=query_gray, 
                        im2=map_gray, 
                        points1=query_points_2D,
                        points2=map_points_2D,
                        title1="From Query", title2="To Map")
    
    print(f"From query: \n{query_points_2D}")
    print(f"to map: \n{map_points_2D}")
    print(f"Map points shift: \n{map_points_2D-valid_map_points[::10,:]}")
    

    plt.show(block=True)