import bpy
from mathutils import Matrix, Vector
import sys
import math
import numpy as np
import addon_utils
import bmesh
import imageio
import cv2
import mathutils
import json
import os
import glob
import shutil
from utils.transform_utils import get_rot_tuple
from PIL import Image
import tempfile

## Notes on reference frames ###
# The drone reference frame (D) is defined as an aircraft body reference frame:
# Z - Yaw axis (pointing downwards)
# Y - Pitch axis (pointing towards the right wing)
# X - Roll axis (pointing towards the the cockpit)


# The camera body frame (B) is initially aligned with the Blender world reference frame and defined as follows:
# Z axis (pointing towards the sky)
# Y axis (pointing towards the direction of motion)
# X axis (pointing rightward)
# The camera boresight axis is pointing along -Z
# The Euler sequence is: 3 1 2 (rot_Z, rot_X, rot_Y)
# The corresponding Blender Euler mode is: YXZ
# With rotation angles [0, 0, 0] the camera boresight axis is pointing downward
# rot_Z = - Yaw
# rot_X = Pitch
# rot Y = Roll

# The camera frame (C) has origin in the camera body frame (B):
# it is defined by applying a rotation of 180 deg to B frame, around its X axis

##


############### HiRISE terrain utilities #####################################
def get_dtm_metadata(terrain_obj, print_data=False):

    # if obj.get("IS_TERRAIN", False):
    dtm_metadata = {
        'curr_res': terrain_obj.get('DTM_RESOLUTION', None),
        'mesh_scale': terrain_obj.get('MESH_SCALE', None),
        'map_scale': terrain_obj.get('MAP_SCALE', None),
        'map_size': terrain_obj.get('MAP_SIZE', None),
        'min_lat': terrain_obj.get('MINIMUM_LATITUDE', None),
        'max_lat': terrain_obj.get('MAXIMUM_LATITUDE', None),
        'east_lon': terrain_obj.get('EASTERNMOST_LONGITUDE', None),
        'west_lon': terrain_obj.get('WESTERNMOST_LONGITUDE', None)
    }

    # Print
    if print_data:
        print("\n--------------- DTM Metadata ---------------")
        print(f"Current Resolution: {dtm_metadata['curr_res']*100} %")
        print(f"Current Scale: {dtm_metadata['mesh_scale']} m/post")
        print(f"Original Scale: {dtm_metadata['map_scale']} m/post")
        print(f"Map Size (w,h): ({dtm_metadata['map_size'][1]},{dtm_metadata['map_size'][0]}) m")
        print(f"Minimum Latitude: {dtm_metadata['min_lat']} deg")
        print(f"Maximum Latitude: {dtm_metadata['max_lat']} deg")
        print(f"Easternmost Longitude: {dtm_metadata['east_lon']} deg")
        print(f"Westernmost Longitude: {dtm_metadata['west_lon']} deg")

    return dtm_metadata


############### Blender import utiliities ####################################

def install_addon(name, path):
    # Check if the add-on is installed
    installed_addons = {mod.module for mod in bpy.context.preferences.addons}
    if name not in installed_addons:
        bpy.ops.preferences.addon_install(filepath=path, overwrite=True)
        # Check if the add-on is enabled
    is_enabled, is_loaded = addon_utils.check(name)
    if not is_enabled:
        bpy.ops.preferences.addon_enable(module=name)
    print(f"Add-on '{name}' is installed and {'enabled' if is_enabled or not is_enabled and bpy.ops.preferences.addon_enable(module=name) else 'not enabled'}")

def load_blend_file(myBlendPath):
    # open ___.blend file
    try:
        bpy.ops.wm.open_mainfile(filepath=myBlendPath)
        print("File opened!")
    except FileNotFoundError:
        print("ERROR: File not found!")

############# Blender environment building utilities ########################
        
def delete_objects(ignore_list=[]):
    for obj in bpy.data.objects:
        if obj.name in ignore_list:
            continue
        obj.select_set(True)
    bpy.ops.object.delete()

def rename_object(name):
    # Called right after an object is added
    # bpy.context.selected_objects should contain a single object
    for i, obj in enumerate(bpy.context.selected_objects):
        #print("Selected objects name: ", i, obj.name)
        if len(bpy.context.selected_objects) == 1:
            obj.name = name
        else:
            raise Exception("More than one objects selected in context!")
    return obj

def set_cycles(scene, cycles_config, print_data=True):
    
    # scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = cycles_config['device']

    cycles = scene.cycles
    # easiest way to remove grainy noise is to increase number of samples
    cycles.samples = cycles_config['samples']
    cycles.use_adaptive_sampling = cycles_config['use_adaptive_sampling']

    cycles.use_progressive_refine = cycles_config['use_progressive_refine']
    #if n_samples is not None:
    #    cycles.samples = n_samples
    cycles.max_bounces = cycles_config['max_bounces']
    cycles.min_bounces = cycles_config['min_bounces']
    cycles.caustics_reflective = cycles_config['caustics_reflective']
    cycles.caustics_refractive = cycles_config['caustics_refractive']
    cycles.diffuse_bounces = cycles_config['diffuse_bounces']
    cycles.glossy_bounces = cycles_config['glossy_bounces']
    cycles.transmission_bounces = cycles_config['transmission_bounces']
    cycles.volume_bounces = cycles_config['volume_bounces']
    cycles.transparent_min_bounces = cycles_config['transparent_min_bounces']
    cycles.transparent_max_bounces = cycles_config['transparent_max_bounces']

    # Avoid grainy renderings (fireflies)
    world = bpy.data.worlds['World']
    world.cycles.sample_as_light = True
    cycles.blur_glossy = cycles_config['blur_glossy']
    cycles.sample_clamp_indirect = cycles_config['sample_clamp_indirect']

    # Ensure no background node
    world.use_nodes = True
    try:
        world.node_tree.nodes.remove(world.node_tree.nodes['Background'])
    except KeyError:
        pass

    if print_data:
        print(f"\n--------------- Set Rendering Engine for Scene {scene.name} ---------------")
        print(f"Rendering engine: {scene.render.engine}")
        print(f"Device: {scene.cycles.device}")
        print(f"Samples: {scene.cycles.samples}")
        print(f"Adaptive sampling: {cycles.use_adaptive_sampling}")
        print(f"Use progressive refine: {cycles.use_progressive_refine}")
        print(f"Bounces (min, max): ({cycles.min_bounces} , {cycles.max_bounces})")
        print(f"Caustics Reflective: {cycles.caustics_reflective}")
        print(f"Caustics Refractive:: {cycles.caustics_refractive}")
        print(f"Diffuse Bounces: {cycles.diffuse_bounces}")
        print(f"Glossy Bounces: {cycles.glossy_bounces}")
        print(f"Transmission Bounces: {cycles.transmission_bounces}")
        print(f"Volume Bounces: {cycles.volume_bounces}")
        print(f"Transparent ounces (min, max): ({cycles.transparent_min_bounces} , {cycles.transparent_max_bounces})")


def set_scene(scene, cam_config, world_config, print_data=True):
    
    
    # Check if the scene exists
    if scene.name not in bpy.data.scenes:
        raise(f"'{scene.name}' scene not found.")
    
    bpy.context.window.scene = scene

    scene.view_settings.exposure = world_config['exposure']
    scene.display_settings.display_device = 'sRGB'
    # #scene.view_settings.view_transform = 'Filmic'

    # # World Lighting parameters
    # world = bpy.data.worlds['World']
    # world.light_settings.use_ambient_occlusion = options['ambient_occlusion']
    # world.light_settings.ao_factor = options['ambient_occlusion_factor']

    scene.render.resolution_x = cam_config["img_width"]
    scene.render.resolution_y = cam_config["img_height"]
    scene.render.pixel_aspect_x = cam_config["aspect_x"]
    scene.render.pixel_aspect_y = cam_config["aspect_y"]
    scene.render.resolution_percentage = 100
    scene.render.use_file_extension = True
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB' # BW
    scene.render.image_settings.color_depth = '8' # default: 8


    if print_data:
        print(f"\n--------------- Set '{scene.name}' Scene ---------------")
        print(f"Resolution: {cam_config['img_width']}x{cam_config['img_height']}")
        print(f"Image aspect: {cam_config['aspect_x']/cam_config['aspect_y']}")
        print(f"File format: {scene.render.image_settings.file_format}")
        print(f"Color Mode: {scene.render.image_settings.color_mode}")
        print(f"Color Depth: {scene.render.image_settings.color_depth}")
        # Setup nodes in a way that respects existing setups if already configured
    set_compositing_nodes(scene, print_data)
        

def set_compositing_nodes(scene, print_data=False):  
    scene.use_nodes = True
    scene.render.use_compositing = True
    scene.view_layers[0].use_pass_z = True
    
    tree = scene.node_tree
    links = tree.links

    # Clear existing nodes
    for node in tree.nodes:
        tree.nodes.remove(node)

    # Create new nodes
    rl = tree.nodes.new('CompositorNodeRLayers')
    # normalize = tree.nodes.new('CompositorNodeNormalize')
    # map_value = tree.nodes.new('CompositorNodeMapValue')
    # map_value.size[0]= 1.0  
    # map_value.use_min = True
    # map_value.min[0] = 0.0  # Optional: Set the minimum value
    # map_value.use_max = True
    # map_value.max[0] = 1.0  # Optional: Set the maximum value

    # For viewer layer
    # vl = tree.nodes.new('CompositorNodeViewer')
    # vl.use_alpha = False

    cl = tree.nodes.new('CompositorNodeOutputFile')
    cl.format.file_format = 'OPEN_EXR'
    cl.format.color_depth = '32'
    cl.base_path = "./tmp/"
    cl.file_slots[0].path = f"{scene.name}_depth"
    

    # links.new(rl.outputs[2], vl.inputs[0]) # link Z to output (for viewer layer)
    links.new(rl.outputs[2], cl.inputs[0]) # link Z to output (for output file layer)

    # links.new(rl.outputs[2], normalize.inputs['Value'])
    # links.new(normalize.outputs['Value'], map_value.inputs['Value'])
    # links.new(map_value.outputs['Value'],cl.inputs[0])

    if print_data:
        print(f"---- Compositing Nodes ----")
        for node in tree.nodes:
            print(f"Node: {node.name}, Type: {node.bl_idname}")

# def set_compositing_nodes_with_mist(scene, options=None, print_data=False):  
#     scene.use_nodes = True
#     scene.render.use_compositing = True
#     if scene.name == "Map":
#         # Enable Z pass
#         scene.view_layers[0].use_pass_z = True # Depth is measured from the camera plane
#     elif scene.name == "Query":
#         # Enable Mist pass and set Mist
#         if options is None:
#             raise("Please provide an option file containing Mist Pass Start and Depth values.")
#         scene.view_layers[0].use_pass_mist = True # Depth is measured from the camera origin 
#         scene.world.mist_settings.start = options["clip_start"]
#         scene.world.mist_settings.depth = options["clip_end"] - options["clip_start"]
#         scene.world.mist_settings.falloff = 'LINEAR'
#     else:
#         raise(f"'{scene.name}' not recognized")
    
#     tree = scene.node_tree
#     links = tree.links

#     # Clear existing nodes
#     for node in tree.nodes:
#         tree.nodes.remove(node)

#     # Create new nodes
#     rl = tree.nodes.new('CompositorNodeRLayers')
#     # vl = tree.nodes.new('CompositorNodeViewer')
#     # vl.use_alpha = False
#     cl = tree.nodes.new('CompositorNodeOutputFile')
#     cl.format.file_format = 'OPEN_EXR'
#     cl.base_path = "./tmp/"
#     cl.file_slots[0].path = f"{scene.name}_depth"
    
#     if scene.name == "Map":
#         # links.new(rl.outputs[2], vl.inputs[0]) # link Z to output
#         links.new(rl.outputs[2], cl.inputs[0]) # link Z to output
#     else:
#         ml = tree.nodes.new('CompositorNodeMath')
#         ml.operation = 'MULTIPLY_ADD'
#         ml.inputs[1].default_value = options["clip_end"] - options["clip_start"]
#         ml.inputs[2].default_value.data.lens = options["clip_start"]
#         links.new(rl.outputs[2], ml.inputs[0]) # link Mist pass to MultiplyAdd node
#         links.new(ml.outputs[0], cl.inputs[0]) # link MultiplyAdd output to Output node


#     if print_data:
#         print(f"---- Compositing Nodes ----")
#         for node in tree.nodes:
#             print(f"Node: {node.name}, Type: {node.bl_idname}")

############# Blender Rendering utilities ########################

def render(scene, cam, filepath=None):

    # Each scene may contain different cameras with same output params (e.g. scene resolution)
    
    # # Directly associate cameras with their respective scenes
    # scene_dict = {
    #     "QueryCam": "Query",
    #     "MapCam": "Map"
    # }

    # # Get the scene name from the dict
    # scene_name = scene_dict.get(cam.name)
  

    bpy.context.window.scene = scene # Set the active scene to the scene found
    scene.camera = cam # Set up camera render for the identified scene
    if filepath is not None:
        scene.render.filepath = filepath + ".png"
        write = True
    else:
        write = False

    # Print the resolution of the scene being rendered
    res_x = scene.render.resolution_x
    res_y = scene.render.resolution_y
    
    print(f"\n--------------- Rendering '{scene.name}' scene ---------------")
    print(f"Camera: '{cam.name}'")
    print(f"Resolution: {res_x}x{res_y}")
    print(f"Rendering image...")

    # Update scene and render 
    bpy.context.view_layer.update()
    bpy.ops.render.render(scene=scene.name, write_still=write) # Render the scene

def render_to_file(scene_obj, cam_obj, store_dir, img_id=0):
    
    # Check if scene exists
    if scene_obj.name not in bpy.data.scenes:
        raise(f"'{scene_obj.name}' scene does not exist.")
    
    # Check if camera exists
    if cam_obj.name not in bpy.data.objects:
        raise(f"'{cam_obj.name}' camera does not exist.")
    
    # Check if images directory exist
    img_dir = store_dir + "/images/"
    if not os.path.isdir(img_dir):
        os.makedirs(img_dir)
    
    # Fine the scene output layer to assign the tile depth path 
    tree = scene_obj.node_tree
    ol = tree.nodes['File Output']
    ol.base_path = f"{store_dir}/exr_files" # set the storage directory for the depth .EXR files
    ol.file_slots[0].path =f"{scene_obj.name}_depth"
    # cam_obj.location = cam_loc
    # cam_obj.rotation_euler = cam_rot

    img_path = img_dir + cam_obj.name + "_%04i" % img_id
    render(scene=scene_obj, cam=cam_obj, filepath=img_path)

def render_tiles_to_file(scene_obj, cam_obj, store_dir, img_id=0, tiles_x=1, tiles_y=1):
    
    # Check if scene exists
    if scene_obj.name not in bpy.data.scenes:
        raise(f"'{scene_obj.name}' scene does not exist.")
    
    # Check if camera exists
    if cam_obj.name not in bpy.data.objects:
        raise(f"'{cam_obj.name}' camera does not exist.")
    
    # Check if images directory exist
    img_dir = store_dir + "/images/"
    if not os.path.isdir(img_dir):
            os.makedirs(img_dir)
    
    scene_obj.render.use_border = True
    scene_obj.render.use_crop_to_border = True

    # # Create a temporary directory to store the tiles
    # with tempfile.TemporaryDirectory() as temp_dir:
    #     tile_paths = []

    img_path = img_dir + cam_obj.name + "_%04i" % img_id


    # Fine the scene output layer to assign the tile depth path 
    tree = scene_obj.node_tree
    ol = tree.nodes['File Output']
    ol.base_path = f"{store_dir}/exr_files"
    depth_base_filename = ol.file_slots[0].path

    # Render each tile and store the file paths
    tile_paths = []
    for y in range(tiles_y):
        for x in range(tiles_x):
            scene_obj.render.border_min_x = x / tiles_x
            scene_obj.render.border_max_x = (x + 1) / tiles_x
            scene_obj.render.border_min_y = y / tiles_y
            scene_obj.render.border_max_y = (y + 1) / tiles_y
            tile_path = img_path + f"_tile_{x}_{y}"  

            # Set tile depth path
            tile_depth_filename = depth_base_filename + f"_tile_{x}_{y}_"
            ol.file_slots[0].path = tile_depth_filename

            render(scene=scene_obj, cam=cam_obj, filepath=tile_path)
            tile_paths.append((tile_path + ".png", x, y))


    # Combine the tiles into a single image
    overall_width = scene_obj.render.resolution_x
    overall_height = scene_obj.render.resolution_y
    final_image = Image.new('RGB', (overall_width, overall_height))

    # Paste each tile into the final image
    tile_width = overall_width // tiles_x
    tile_height = overall_height // tiles_y
    for tile_path, x, y in tile_paths:
        tile = Image.open(tile_path)
        final_image.paste(tile, (x * tile_width, (tiles_y - y - 1) * tile_height))

    # Save the final combined image
    final_image.save(img_path + ".png")

    print(f"Final image saved to: {img_path + '.png'}")


################## Blender Lighting #####################

# def update_lighting(config):
#     light = bpy.data.objects["Light"]
#     light.energy = config['energy'] # W/m^2, vs. 1361.0 W/m^2 on earth: 1/25
#     light.angle = config['angular_diameter']

#     light = bpy.data.objects["Light"]
#     light.location = config["translation"]
#     light.rotation_mode = 'XYZ'
#     light.rotation_euler = get_rot_tuple(config['rotation'])


def add_light_source(light_type, config, name):
    # https://docs.blender.org/api/current/bpy.ops.object.html#bpy.ops.object.light_add
    # light_type choices: "SPOT", "SUN", "SPOT", "AREA"

    bpy.ops.object.light_add(type=light_type)
    # # light = rename_object(name)
    light = bpy.data.objects[light_type.capitalize()]

    if name is not None:
        light.name = name


    light.rotation_mode = 'XYZ'
    light.location = config['translation']
    light.rotation_euler = get_rot_tuple(config['rotation'])

    light.data.energy = config['energy'] # W/m^2, vs. 1361.0 W/m^2 on earth: 1/25
    light.data.angle = config['angular_diameter']*np.pi/180 # needs the value in rad
    light.visible_glossy = config['glossy'] # Enable/disable the glossy option in Ray Visibility  (object tab)

    return light

################## Blender Cameras utilities #####################

def add_camera(cam_config, xyz=(0, 0, 0), rot_vec_rad=(0, 0, 0), name=None):
    
    # The camera object frame (O) is aligned with Blender reference frame and defined as follows:
    # Z - Yaw axis (pointing upward)
    # Y - Roll axis (pointing towards the direction of motion)
    # X - Pitch axis (pointing rightward)
    # The camera boresight axis is pointing along -Z
    # The Euler sequence is: 3 1 2 (Yaw, Pitch, Roll)
    # The corresponding Blender Euler mode is: YXZ
    # With rotation angles [0, 0, 0] the camera boresight axis is pointing downward

    # The camera frame (C) has origin in the camera object frame (O):
    # it is defined by applying a rotation of 180 deg to O frame, around its X axis

    #bpy.ops.object.camera_add(enter_editmode=False, align='VIEW')
    bpy.ops.object.camera_add()
    # cam = bpy.context.object
    cam = bpy.data.objects['Camera']
    #cam = bpy.context.active_object

    if name is not None:
        cam.name = name

    cam.rotation_mode = 'YXZ'
    cam.location = xyz
    cam.rotation_euler = rot_vec_rad

    cam.data.type = cam_config['proj_model']
    if cam_config['proj_model'] == 'PERSP':
        cam.data.lens = cam_config['focal_length']
        cam.data.shift_x = cam_config['shift_x']
        cam.data.shift_y = cam_config['shift_y']
        cam.data.sensor_fit = cam_config['sensor_fit']
        cam.data.sensor_width = cam_config['sensor_width']    
    elif cam_config['proj_model'] == 'ORTHO':
        cam.data.ortho_scale = cam_config["ortho_scale"]
        cam.data.sensor_fit = cam_config['sensor_fit']
    else:
        raise("Projective model not supported. Select 'PERSP' or 'ORTHO'.")
    
    # cam.data.sensor_height = sensor_height
    cam.data.clip_start = cam_config['clip_start']
    cam.data.clip_end = cam_config['clip_end']
    return cam


def set_camera_above_terrain(camera, terrain, x, y, altitude):
    
    if not camera:
        print(f"Camera '{camera.name}' not found.")
        return
    
    if not terrain:
        print(f"Terrain '{terrain.name}' not found.")
        return

    # Create a BMesh from the terrain's mesh data
    bm = bmesh.new()
    bm.from_mesh(terrain.data)
    
    # Transform the BMesh to apply any object transformations (location, rotation, scale)
    bm.transform(terrain.matrix_world)
    
    # Create a BVHTree from BMesh
    bvh = mathutils.bvhtree.BVHTree.FromBMesh(bm)

    # Calculate the ray direction (downward in the local Z direction)
    ray_direction = Vector((0, 0, -1))
    
    # Starting point for the ray is a bit above the terrain to ensure it starts outside
    ray_origin = Vector((x, y, 100000))  # Large Z value to start high above

    # Cast the ray from above to the terrain
    location, normal, index, distance = bvh.ray_cast(ray_origin, ray_direction)

    if location is None:
        print("Ray cast did not hit the terrain")
        return

    # Set the camera's location
    # Z coordinate is the hit location's Z plus the desired altitude
    camera.location = (x, y, location.z + altitude)
    
    # Cleanup BMesh and BVHTree
    bm.free()






