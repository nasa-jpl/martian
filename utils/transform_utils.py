import math
import numpy as np


######### Math utilities ###################
def get_rot_tuple(tuple_deg):
    # Convers tuple from degrees to radians
    return ( math.radians(tuple_deg[0]), 
             math.radians(tuple_deg[1]), 
             math.radians(tuple_deg[2]) )

def XtoSO3(angle):
    '''  Convert rotation angle around X to SO3 rotation matrix
    angle [rad] - rotation angle
    '''
    cos = np.cos(angle)
    sin = np.sin(angle)

    R =  np.array((
        (1,     0,     0),
        (0,  cos,    -sin),
        (0,  sin,     cos)
    ), dtype=np.float64)

    # print(F"X_inv - X_transp: {np.linalg.inv(R) - R.transpose()}")

    return np.asmatrix(R)

def YtoSO3(angle):
    '''  Convert rotation angle around Y to SO3 rotation matrix
    angle [rad] - rotation angle
    '''
    cos = np.cos(angle)
    sin = np.sin(angle)

    R =  np.array((
        (cos,    0,   sin),
        (0,      1,     0),
        (-sin,   0,   cos)
    ), dtype=np.float64)

    # print(F"Y_inv - Y_transp: {np.linalg.inv(R) - R.transpose()}")
    return np.asmatrix(R)

def ZtoSO3(angle):
    '''  Convert rotation angle around Z to SO3 rotation matrix
    angle [rad] - rotation angle
    '''
    cos = np.cos(angle)
    sin = np.sin(angle)

    R =  np.array((
        (cos,  -sin,     0),
        (sin,   cos,     0),
        (0,       0,     1)
    ), dtype=np.float64)

    # print(F"Z_inv - Z_transp: {np.linalg.inv(R) - R.transpose()}")

    return np.asmatrix(R)

def Eul312toSO3(angle_X, angle_Y, angle_Z):
    '''  Convert rot_X, rot_Y, rot_Z angles to SO3 matrix according to rotation order 312: R(rot_Y)*R(rot_X)*R(rot_Z) 
    angle_Z - Around Z axis (pointing upward in Blender object frame) [rad]
    angle_Y - Around Y axis (pointing in the direction of motion in Blender object frame) [rad]
    angle_X - Around X axis (pointing rightward w.r.t direction of motion) [rad]
    
    
    '''
    R =  np.dot(ZtoSO3(angle_Z), np.dot(XtoSO3(angle_X), YtoSO3(angle_Y)) )

    # print(F"R_inv - R_transp: {np.linalg.inv(R) - R.transpose()}")

    return R

##### Blender 2D transformations


def img2blender(point, size, pixel_res):

    # print(f"point: {point}")
    # Transformation from image frame to camera frame
    width, height = size
    img_center = np.array([width/2, height/2])
    point_cam = (point - img_center)*pixel_res
    # print(f"point_cam: {point_cam}")
    # Transformation from camera frame to blender reference frame
    angle = np.pi
    cos = np.cos(angle)
    sin = np.sin(angle)

    R =  np.array((
        (1,     0),
        (0,  cos)
    ), dtype=np.float64)
    point_blender = np.dot(R, point_cam) # (2,)
    
    # print(f"point_blend: {point_blender}")

    return point_blender

def blender2img(point_blender, size, pixel_res):

    width, height = size
    # Transformation from blender frame to camera reference frame
    angle = np.pi
    cos = np.cos(angle)
    sin = np.sin(angle)
    R =  np.array(( # avoid using np.asmatrix in np.asmatrix(np.array(...))
        (1,     0),
        (0,  cos)
    ), dtype=np.float64)
    point_cam = np.dot(R, point_blender) 

    # Transformation from image frame to camera frame
    point = point_cam/pixel_res + np.array([width/2, height/2]) # (2,)
    return point
