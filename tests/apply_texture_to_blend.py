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
from blender_addons.hirise_dtmimporter.dtmimporter.mesh.terrain import BTerrain
from blender_addons.hirise_dtmimporter.dtmimporter.mesh.dtm import DTM
from bpy.types import ShaderNodeTexImage

def delete_all_objects():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=".")

    parser.add_argument('--in_blend_filepath', type=str, default='hirise_assets/jezero_crater/jezero_HP.blend', help="Path to the input .blend file.")
    parser.add_argument('--ortho_filepath', type=str, default='hirise_assets/jezero_crater/ortho-images/ESP_046060_1985_RED_A_01_ORTHO.JP2', help="Path to the ortho image to apply as textures on top of DTM.")
    parser.add_argument('--out_blend_dest', type=str, default='hirise_assets/jezero_crater/blend_files', help="Path to the repo storing the .blend file.")
    parser.add_argument('--out_blend_filename', type=str, default='jezero_50resDTM_A01ortho.blend', help="Output .blend file name.")
    parser.add_argument('--keep_blender_running', default=False, action='store_true', help="If enabled, kills blender process after rendering." )
    

    print("sys.argv:", sys.argv)

    if '--' not in sys.argv:
        argv = []
    else:
        argv = sys.argv[sys.argv.index('--') + 1:]

    print("argv passed to parser:", argv)
    args = parser.parse_args(argv)
    print("Parsed arguments:", args)

    
    # Install HiRISE DTM importer
    addon_name = "dtmimporter"
    addon_path = f"./blender_addons/hirise_dtmimporter/{addon_name}.zip"
    utils.install_addon(addon_name, addon_path)

    # Check if the blend file directory exists, and create it if it doesn't
    if not os.path.exists(args.out_blend_dest):
        os.makedirs(args.out_blend_dest)
        print(f"Created directory: {args.out_blend_dest}")


    # Load blend file
    utils.load_blend_file(args.in_blend_filepath)
    for obj in bpy.data.objects:
        print(obj.name)

    # Dele all the objects except the terrain
    bpy.ops.object.select_all(action='DESELECT')
    terrain = bpy.data.objects.get("Terrain")
    bpy.ops.object.select_all(action='SELECT')
    if terrain is not None:
        terrain.select_set(False)
        bpy.ops.object.delete()
        terrain.select_set(True)
    else:
        raise("Object 'Terrain' not found.")

    
    # Setup Blender UI
    scene = bpy.context.scene
        # Set correct units
    scene.unit_settings.system = 'METRIC'
    scene.unit_settings.scale_length = 1.0
        # Set clip end view in all 3D view areas
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.clip_end = 100000

    ## Apply material to the terrain
    # Clear terrain materials
    terrain.data.materials.clear()

    # Create a new material
    material = bpy.data.materials.new(name="Mars")
    terrain.data.materials.append(material)
    
    
    # Enable 'Use Nodes'
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    # Clear default nodes
    for node in nodes:
        nodes.remove(node)

    # Add a Principled BSDF node
    principled_bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    principled_bsdf.location = (0, 0)

    # Add an Image Texture node
    image_texture = material.node_tree.nodes.new(type="ShaderNodeTexImage")
    image_texture.location = (-300, 0)
    
    # Load the image file
    abs_path = os.path.abspath(args.ortho_filepath)
    dir_abs_path = os.path.dirname(abs_path)
    file_name = os.path.basename(abs_path)
    print(f"//{file_name}")
    print(f"{dir_abs_path}/",)
    print(f"{file_name}")
    bpy.ops.image.open(filepath=f"//{file_name}",
                    directory=f"{dir_abs_path}/",
                    files=[{"name": f"{file_name}"}],
                    show_multiview=False)

    # bpy.ops.image.open(filepath="//ESP_046060_1985_RED_C_01_ORTHO.JP2",
    #                 directory="/home/dpisanti/LORNA/mars-sim/hirise_assets/jezero_crater/ortho-images/",
    #                 files=[{"name": "ESP_046060_1985_RED_C_01_ORTHO.JP2"}],
    #                 show_multiview=False)
                    
    # Determine the name assigned to the image in Blender and set the image source to 'FILE'
    image_name = bpy.data.images[0].name
    bpy.data.images[image_name].source = 'FILE'
    bpy.data.images[image_name].colorspace_settings.name = 'sRGB'

    # Assign the loaded image to the Image Texture node
    image_texture.image = bpy.data.images[image_name]

    # Add a Texture Coordinate node
    texture_coordinate = material.node_tree.nodes.new(type="ShaderNodeTexCoord")
    texture_coordinate.location = (-600, 0)

    # Add links
    # Link the Texture Coordinate node to the Image Texture node
    material.node_tree.links.new(texture_coordinate.outputs['Generated'], image_texture.inputs['Vector'])
    # Link the Image Texture node to the Principled BSDF node
    material.node_tree.links.new(image_texture.outputs['Color'], principled_bsdf.inputs['Base Color'])

    # Add a Material Output node
    material_output = nodes.new(type="ShaderNodeOutputMaterial")
    material_output.location = (200, 0)

    # Link the Principled BSDF node to the Material Output node
    links.new(principled_bsdf.outputs['BSDF'], material_output.inputs['Surface'])


    # Save the blend file
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(args.out_blend_dest, args.out_blend_filename))
    print(f"DTM resolution: {terrain.dtm_resolution}")

    # # Reload terrain with new resolution percentage
    # scaled_dtm_res = 0.2 # (fraction of the original DTM resolution)
    # terrain = bpy.data.objects.get(env_config['Terrain']["terrain_name"])
    # terrain.dtm_resolution = scaled_dtm_res*100
    # dtm_path = env_config['Terrain']["dtm_file"]
    # dtm = DTM(dtm_path, scaled_dtm_res)
    # BTerrain.reload(terrain, dtm)
    # bpy.ops.wm.save_as_mainfile(filepath=f"hirise_assets/jezero_crater/jezero_{int(scaled_dtm_res*100)}res_wo_objects.blend")
    # print(f"DTM resolution: {terrain.dtm_resolution}")

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
