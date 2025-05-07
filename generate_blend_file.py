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
import imageio
import cv2
from utils.transform_utils import get_rot_tuple
from utils.blend_utils import get_dtm_metadata, install_addon
from configs import Config
from blender_addons.hirise_dtmimporter.dtmimporter.mesh.terrain import BTerrain
from blender_addons.hirise_dtmimporter.dtmimporter.mesh.dtm import DTM
from bpy.types import ShaderNodeTexImage

def delete_all_objects():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=".")

    parser.add_argument('--main_yaml', type=str, default='./configs/jezero.yaml', help="Min yaml file with environment configuration")
    parser.add_argument('--dtm_resolution', type=float, default=10., help="Percentage scale for terrain model resolution. 100\% loads the "
                                                                            "model at full resolution (i.e. one vertex for each post in the "
                                                                            "original terrain model) and is *MEMORY INTENSIVE*. Downsampling "
                                                                            "uses Nearest Neighbors. The downsampling algorithm may need to "
                                                                            "alter the resolution you specify here to ensure it results in a "
                                                                            "whole number of vertices. If it needs to alter the value you "
                                                                            "specify, you are guaranteed that it will shrink it (i.e. "
                                                                            "decrease the DTM resolution.")
    parser.add_argument('--blend_dest', type=str, default='./blend_files', help="Path to the repo storing the .blend file.")
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
    install_addon(addon_name, addon_path)

    # Check if the blend file directory exists, and create it if it doesn't
    if not os.path.exists(args.blend_dest):
        os.makedirs(args.blend_dest)
        print(f"Created directory: {args.blend_dest}")

    # Dele all the objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # load config
    config_manager = Config(BASE_PATH + 'configs/'+ args.main_yaml)
    config = config_manager.get_config()
    dtm_filepath = config["Terrain"]["dtm_file"]
    ortho_filepath = config["Terrain"]["high_res_ortho"]

    # Load the terrain with default resolution of 10% (this makes things easier)
    print(f"\n...loading Terrain with default resolution (10%)")
    scaled_dtm_res_default = 0.10
    dtm = DTM(dtm_filepath, scaled_dtm_res_default)
    BTerrain.new(dtm)
    print(f"...complete!")
    
    # Setup Blender UI
    print(f"\n...setting Blender UI")
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
    print(f"...complete!")

    ## Apply material to the terrain
    print(f"\n... adding new material to Terrain")
    terrain = bpy.data.objects.get("Terrain") # Get the terrain object
    if terrain is not None:
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
        print(f"/...load ortho-image")
        abs_path = os.path.abspath(ortho_filepath)
        dir_abs_path = os.path.dirname(abs_path)
        file_name = os.path.basename(abs_path)
        print(f"image abs path: {abs_path}")
        # print(f"//{file_name}")
        # print(f"{dir_abs_path}/",)
        # print(f"{file_name}")
        bpy.ops.image.open(filepath=abs_path)
        print(f"/...complete!")
        # bpy.ops.image.open(filepath=f"//{file_name}",
        #                 directory=f"{dir_abs_path}/",
        #                 files=[{"name": f"{file_name}"}],
        #                 show_multiview=False)

        # bpy.ops.image.open(filepath="//ESP_046060_1985_RED_C_01_ORTHO.JP2",
        #                 directory="/home/dpisanti/LORNA/mars-sim/hirise_assets/jezero_crater/ortho-images/",
        #                 files=[{"name": "ESP_046060_1985_RED_C_01_ORTHO.JP2"}],
        #                 show_multiview=False)
                        
        # Determine the name assigned to the image in Blender and set the image source to 'FILE'
        print("/...applying textures")
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

        # Set material roughness
        principled_bsdf.inputs["Roughness"].default_value = config["Terrain"]["roughness"]

        # Add a Material Output node
        material_output = nodes.new(type="ShaderNodeOutputMaterial")
        material_output.location = (200, 0)

        # Link the Principled BSDF node to the Material Output node
        links.new(principled_bsdf.outputs['BSDF'], material_output.inputs['Surface'])
        print("/...complete!")
        print("A material with ortho-image has texture has been applied to Terrain")
    else:
        print("Object 'Terrain' not found.")


    # Reload terrain with new resolution percentage
    print(f"...load Terrain with { args.dtm_resolution}% resolution")
    scaled_dtm_res = args.dtm_resolution/100 # (fraction of the original DTM resolution)
    if scaled_dtm_res != scaled_dtm_res_default:
        terrain.dtm_resolution =args.dtm_resolution
        dtm = DTM(dtm_filepath, scaled_dtm_res)
        BTerrain.reload(terrain, dtm)
    print(f"...complete.")
    # bpy.ops.wm.save_as_mainfile(filepath=f"hirise_assets/jezero_crater/jezero_{int(scaled_dtm_res*100)}res_wo_objects.blend")
    


    # Save the blend file
    yaml_name = os.path.splitext(os.path.basename(args.main_yaml))[0]
    blend_filename = f"{yaml_name}_{int(args.dtm_resolution)}res"
    blend_file_path = os.path.join(args.blend_dest, blend_filename) + ".blend"
    # blend_file_path = args.blend_dest
    print(f"\n...saving the .blend file to: {blend_file_path}")
    try:
        bpy.ops.wm.save_mainfile(filepath=blend_file_path)
        # bpy.ops.wm.save_as_mainfile(filepath=blend_file_path)
        print(f"...complete.")
    except Exception as e:
        print(f"Failed to save blend file: {e}")
        sys.exit(1)
    print(f"DTM resolution: {terrain.dtm_resolution}")
    print(f"File saved at {blend_file_path}")

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
