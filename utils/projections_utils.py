
import numpy as np
import matplotlib.pyplot as plt
from utils.transform_utils import Eul312toSO3, XtoSO3
from utils.plot_utils import plot_correspondences
from utils.data_utils import load_depth, normalize_depth
import os
# os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"


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

def get_cam2cam_correspondences(cam0_scene, cam0, cam0_gray, cam0_depth, cam1_scene, cam1, cam1_gray, cam1_depth, bound=0, px_sampling_step = 1000, plot_on_dmap=False, dest_dir=None):
   
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
    R_cam0_wc = np.matmul(Eul312toSO3(cam0.rotation_euler.x, cam0.rotation_euler.y, cam0.rotation_euler.z), XtoSO3(np.pi)) # the convention in wc is the other way round
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
    R_cam1_wc = np.matmul(Eul312toSO3(cam1.rotation_euler.x, cam1.rotation_euler.y, cam1.rotation_euler.z), XtoSO3(np.pi)) # the convention in wc is the other way round
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
    cam0_gray = np.array(cam0_gray)
    cam1_gray = np.array(cam1_gray)    
    plot_correspondences(im1=cam0_gray, 
                         im2=cam1_gray, 
                        points1=cam0_points_2D, 
                        points2=cam1_points_2D,
                        title1 = f"From {cam0.name}", title2 = f"To {cam1.name}",
                        filepath=os.path.join(dest_dir, f"from_{cam0.name}_to_{cam1.name}.png") if dest_dir else None)

    plot_correspondences(im1=cam1_gray, 
                         im2=cam0_gray, 
                        points1=cam1_points_2D,
                        points2=cam0_points_2D_reproj,
                        title1=f"From {cam1.name}", title2=f"To {cam0.name}",
                        filepath=os.path.join(dest_dir, f"from_{cam1.name}_to_{cam0.name}.png") if dest_dir else None)
    
    if plot_on_dmap:
        cam0_depth_img = normalize_depth(cam0_depth, cam0)
        cam1_depth_img = normalize_depth(cam1_depth, cam1)
        plot_correspondences(im1=cam0_depth_img, 
                        im2=cam1_depth_img, 
                        points1=cam0_points_2D, 
                        points2=cam1_points_2D,
                        title1 = f"From (Depth) {cam0.name}", title2 = f"To (Depth) {cam1.name}",
                        filepath=os.path.join(dest_dir, f"from_{cam0.name}_to_{cam1.name}_dmap.png") if dest_dir else None)
    
     

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

