import os, signal
import sys
import bpy
import numpy as np
import matplotlib.pyplot as plt
import utils
import imageio
import json
import cv2
from configs import Config
from classes.ortho_image_manager import OrthoImageManager
import utils.blend_utils as blend_utils 
from utils.transform_utils import get_rot_tuple
from utils.projections_utils import get_cam2cam_correspondences
import utils.data_utils as data_utils
from utils.transform_utils import blender2img
from blender_addons.hirise_dtmimporter.dtmimporter.mesh.terrain import BTerrain
from blender_addons.hirise_dtmimporter.dtmimporter.mesh.dtm import DTM
import shutil
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

BASE_PATH = os.getcwd() + "/"
sys.path.append(BASE_PATH)



class BlenderRunner:
    def __init__(self, args):
        self.args = args

    def run(self):
        # Print all arguments to verify they include the mode and common arguments
        print(f"Running in {self.args.mode} mode")
        print(f"Arguments: {vars(self.args)}")

        # Import add-on
        addon_name = "addon_ground_truth_generation"
        addon_path = f"./blender_addons/VisionBlender/{addon_name}.py"
        blend_utils.install_addon(addon_name, addon_path)
            
        # Install HiRISE DTM importer
        addon_name = "dtmimporter"
        addon_path = f"./blender_addons/hirise_dtmimporter/{addon_name}.zip"
        blend_utils.install_addon(addon_name, addon_path)

        self.setup_scenes()
        if self.args.mode == 'demo':
            self.run_demo()
        elif self.args.mode == 'dataset':
            self.run_dataset()

        if not self.args.keep_blender_running:
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

    def generate_blend_file(self, config, dtm_resolution, save_blend=False, blend_dest=None, blend_filename=None):

        # Dele all the objects
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()

        # load config
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
        print(f"...load Terrain with { dtm_resolution}% resolution")
        scaled_dtm_res = dtm_resolution/100 # (fraction of the original DTM resolution)
        if scaled_dtm_res != scaled_dtm_res_default:
            terrain.dtm_resolution =dtm_resolution
            dtm = DTM(dtm_filepath, scaled_dtm_res)
            BTerrain.reload(terrain, dtm)
        print(f"...complete.")

        # Save the blend file
        if save_blend:
            print(f"\n...save the .blend file.")
            bpy.ops.wm.save_as_mainfile(filepath=os.path.join(blend_dest, blend_filename) + ".blend")
            print(f"...complete.")
            print(f"DTM resolution: {dtm_resolution}")
            print(f"File saved at {os.path.join(blend_dest, blend_filename)}.blend")

    def setup_scenes(self):

        # LOAD ENVIRONMENT CONFIGURATION
        # Load the yaml with the enviornment configuration
        self.env_config_manager = Config(BASE_PATH + 'configs/'+ self.args.main_yaml)
        self.env_config = self.env_config_manager.get_config()
        
        # The argument parser checks that only one arg between "--generate_blend_file" and "--blend_file" is provided
        if self.args.generate_blend_file:
            self.generate_blend_file(self.env_config, self.args.dtm_resolution, self.args.save_blend, self.args.blend_dest, self.args.blend_filename)
        else: # it loads the blend file provided as alternative
            # Load the blend file (doesn't work from "blender start")
            blend_path=f"{self.args.blend_file}.blend"
            blend_utils.load_blend_file(blend_path)
            # Set terrain roughness
            bpy.data.materials["Mars"].node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = self.env_config["Terrain"]["roughness"]

        # Import Terrain with metadata
        self.terrain_mesh = bpy.data.objects[self.env_config['Terrain']["terrain_name"]]
        dtm_metadata = blend_utils.get_dtm_metadata(self.terrain_mesh, print_data=True)
        self.map_ortho_width = dtm_metadata['map_size'][1]
        self.map_ortho_height = dtm_metadata['map_size'][0]  # m
        map_px_res = self.args.map_px_res # map pixel resolution [m /pixel] (the texture resolution for jezero.yaml is 1 m/pixel)
        if self.args.mode == "demo":
            if self.args.map_dir:
                # Load cam data
                with open(os.path.join(self.args.map_dir, "cam_data.json")) as f:
                    map_cam_data = json.load(f)
                self.map_ortho_width = map_cam_data['ortho_scale']
                map_width, map_height = map_cam_data['img_width'], map_cam_data['img_height']
                map_px_res = self.map_ortho_width / map_width
                

    
        # LOAD AND SAVE CAMERA CONFIGURATIONS FOR THE SPECIFIC TERRAIN
        # Load the yaml with the defualt camera configuration
        cam0_config_manager = Config(BASE_PATH + 'configs/base_cam/persp_cam.yaml')
        mapcam_config_manager = Config(BASE_PATH + 'configs/base_cam/ortho_cam.yaml')
        # Save the new config file for the specified terrain
        mapcam_config_manager.update_config({
            'ortho_scale':self.map_ortho_width,
            'img_width':int(self.map_ortho_width/map_px_res),
            'img_height':int(self.map_ortho_height/map_px_res)
        })
        cam0_config = cam0_config_manager.get_config()  
        mapcam_config = mapcam_config_manager.get_config()

        # Add light light
        light = blend_utils.add_light_source(light_type="SUN", config=self.env_config['SunLight'], name="SunLight")

        # Generate two linked scenes for map and queries rendering
        cam0_scene_name = "Persp0"
        map_scene_name = "Ortho"  
        bpy.context.scene.name = cam0_scene_name
        bpy.ops.scene.new(type='LINK_COPY')   
        bpy.context.scene.name = map_scene_name
        cam0_scene = bpy.data.scenes[cam0_scene_name]     
        map_scene = bpy.data.scenes[map_scene_name]
        if self.args.mode == "demo": # Set the 2nd perspective camera for demo mode
            # Load config
            cam1_config_manager = Config(BASE_PATH + 'configs/base_cam/persp_cam.yaml')
            cam1_config = cam1_config_manager.get_config()
            # Create scene
            cam1_scene_name = "Persp1"         
            bpy.ops.scene.new(type='LINK_COPY')
            bpy.context.scene.name = cam1_scene_name
            cam1_scene = bpy.data.scenes[cam1_scene_name]
            # Set scene
            cam1_name = "PerspCam1"
            blend_utils.set_scene(scene=cam1_scene, cam_config=cam1_config, world_config=self.env_config['World'])
            cam1 = blend_utils.add_camera(cam1_config, name=cam1_name) # Manually add the cameras
            cam1_scene.collection.objects.link(cam1) # Link to the scene
            # Set attributes
            self.cam1 = cam1
            self.cam1_scene = cam1_scene
            self.cam1_config_manager = cam1_config_manager
            # Set rendering engine
            blend_utils.set_cycles(scene=cam1_scene, cycles_config=self.env_config['Cycles'])

        # Set Cam0 scene
        cam0_name = "PerspCam0"
        blend_utils.set_scene(scene=cam0_scene, cam_config=cam0_config, world_config=self.env_config['World'])
        cam0 = blend_utils.add_camera(cam0_config, name=cam0_name) # Manually add the cameras
        cam0_scene.collection.objects.link(cam0) # Link to the scene
        #   Set rendering engine
        blend_utils.set_cycles(scene=cam0_scene, cycles_config=self.env_config['Cycles'])

        # Set Map scene
        map_cam_name = "OrthoCam"
        blend_utils.set_scene(scene=map_scene, cam_config=mapcam_config, world_config=self.env_config['World'])
        map_cam = blend_utils.add_camera(mapcam_config, name=map_cam_name) # Manually add the cameras
        map_scene.collection.objects.link(map_cam) # Link to the scene
        print(f"Map config resolution: {(mapcam_config['img_width'], mapcam_config['img_height'])}")
        print(f"Map scene resolution: {(map_scene.render.resolution_x, map_scene.render.resolution_y)}")
        #   Set rendering engine
        blend_utils.set_cycles(scene=map_scene, cycles_config=self.env_config['Cycles'])

        # Set class attributes
        self.light = light
        self.map_cam = map_cam
        self.map_scene = map_scene
        self.mapcam_config_manager = mapcam_config_manager
        self.cam0 = cam0
        self.cam0_scene = cam0_scene
        self.cam0_config_manager = cam0_config_manager

        print(f"\nAdded Objects:", bpy.data.objects.keys())

        if self.args.save_blend_scenes:
            print(f"\n...save the scene in a new .blend file.")
            if self.args.blend_file:
                bpy.ops.wm.save_as_mainfile(filepath= self.args.blend_file + "_scenes.blend")
            if self.args.generate_blend_file:
                bpy.ops.wm.save_as_mainfile(filepath=os.path.join(self.args.blend_dest, self.args.blend_filename) + "_scenes.blend")
            print(f"...complete.")
            print(f"File saved at {os.path.join(self.args.blend_dest, self.args.blend_filename)}._scenes.blend")


    def run_demo(self):
        print("\n**************** Running demo ****************************")
        print(f"Running demo with sun azimuth: {self.args.sun_az}, sun elevation: {self.args.sun_el}")
        
        # Set save dir
        save_dir = os.path.join(self.args.dest_dir, self.args.demo_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save config file:
        self.cam0_config_manager.save_config(BASE_PATH + f"configs/{os.path.splitext(self.args.main_yaml)[0]}_persp_cam0.yaml")
        self.cam1_config_manager.save_config(BASE_PATH + f"configs/{os.path.splitext(self.args.main_yaml)[0]}_persp_cam1.yaml")  
        self.mapcam_config_manager.save_config(BASE_PATH + f"configs/{os.path.splitext(self.args.main_yaml)[0]}_ortho_cam.yaml")

        # Set variables
        light = self.light
        map_cam = self.map_cam
        map_scene = self.map_scene
        cam0 = self.cam0
        cam0_scene = self.cam0_scene
        cam1 = self.cam1
        cam1_scene = self.cam1_scene

        # Set Sun position
        light.rotation_euler = get_rot_tuple((0, 90. - self.args.sun_el, self.args.sun_az))
    
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
        blend_utils.set_camera_above_terrain(cam0, self.terrain_mesh, self.args.cam0_loc_x, self.args.cam0_loc_y, self.args.cam0_altitude)  
        cam0_loc = cam0.location
        cam0_rot = [self.args.cam0_pitch, self.args.cam0_roll, self.args.cam0_yaw] # X, Y, Z euler angles in B frame (deg) ([0,0,0] points downwards)
        cam0.rotation_euler = get_rot_tuple(cam0_rot)

        # Set Cam 1 pose
        blend_utils.set_camera_above_terrain(cam1, self.terrain_mesh, self.args.cam1_loc_x, self.args.cam1_loc_y, self.args.cam1_altitude)  
        cam1_loc = cam1.location
        cam1_rot = [self.args.cam1_pitch, self.args.cam1_roll, self.args.cam1_yaw] # X, Y, Z euler angles in B frame (deg) ([0,0,0] points downwards)
        cam1.rotation_euler = get_rot_tuple(cam1_rot)

        # Set Ortho cam pose
        blend_utils.set_camera_above_terrain(map_cam, self.terrain_mesh,  self.env_config["ortho_cam_x"], self.env_config["ortho_cam_y"], self.env_config["ortho_cam_altitude"])  
        map_cam.rotation_euler = (0,0,0)


        # Render perspective camera images and depth
        idx=0
        persp_dir = os.path.join(save_dir, "perspective")
        if not os.path.isdir(persp_dir):
            os.makedirs(persp_dir)
        blend_utils.render(cam0_scene, cam0, f"{persp_dir}/images/{cam0_scene.name}")
        blend_utils.render(cam1_scene, cam1, f"{persp_dir}/images/{cam1_scene.name}")
        data_utils.save_depth(cam0_scene, cam0, persp_dir, idx, save_png=True)
        data_utils.save_depth(cam1_scene, cam1, persp_dir, idx, save_png=True)
        data_utils.save_pose_data(cam0, self.args.cam0_altitude, persp_dir, idx)
        data_utils.save_pose_data(cam1, self.args.cam1_altitude, persp_dir, idx)
        data_utils.save_cam_data(cam0, cam0_scene, persp_dir)
        data_utils.save_cam_data(cam1, cam1_scene, persp_dir)
        data_utils.save_config_from_blender(cam0_scene, light, persp_dir)
        data_utils.save_config_from_blender(cam1_scene, light, persp_dir)
        # Get images and depth for the perspective cameras
        #   Depth
        cam0_depth = data_utils.load_depth(f"{persp_dir}/depth/{cam0.name}" + "_%04i" % idx + "_depth")
        cam1_depth = data_utils.load_depth(f"{persp_dir}/depth/{cam1.name}" + "_%04i" % idx + "_depth")
        #  image
        cam0_gray = cv2.imread(f"{persp_dir}/images/{cam0_scene.name}.png", cv2.IMREAD_GRAYSCALE)
        cam1_gray = cv2.imread(f"{persp_dir}/images/{cam1_scene.name}.png", cv2.IMREAD_GRAYSCALE)

        # Generate perspective cams correspondences
        get_cam2cam_correspondences(cam0_scene, cam0, cam0_gray, cam0_depth, cam1_scene, cam1, cam1_gray, cam1_depth, bound=0, px_sampling_step=100, plot_on_dmap=True, dest_dir=persp_dir)
        # get_cam2cam_correspondences(cam0_scene, cam0, map_scene, map_cam, bound=0, px_sampling_step=1, plot_on_dmap=False)

        if self.args.map_dir is None:
           
            # Map pararams
            map_tiles_x = 8
            map_tiles_y = 16
            map_width  = map_scene.render.resolution_x
            map_height = map_scene.render.resolution_y

            # Set the map storage directory
            map_dir = os.path.join(save_dir, "map")
            if not os.path.isdir(map_dir):
                os.makedirs(map_dir)
            # Set configuration directory
            map_conf_dir = os.path.join(map_dir, "configs")
            if not os.path.isdir(map_conf_dir):
                os.makedirs(map_conf_dir)
            # Save cam data
            data_utils.save_cam_data(map_cam, map_scene, map_dir)
            #   Save config files
            self.env_config_manager.save_config(os.path.join(map_conf_dir, self.args.main_yaml))
            self.mapcam_config_manager.save_config(os.path.join(map_conf_dir,f"mapcam.yaml"))
            #   Save config files from Blender (for a sanity check)
            data_utils.save_config_from_blender(map_scene, light, map_conf_dir)
            # Save map params
            map_params={
                # Path
                "map_path":map_dir, # Map dataset path
                # Resolution:
                "map_px_res":self.args.map_px_res, # m / pixels
                # Tiles
                "tiles_x":map_tiles_x,
                "tiles_y":map_tiles_y
            }
            data_utils.write_json(map_params, os.path.join(map_dir, "params.json")) 

            # Set map output repo
            map_store_dir = os.path.join(map_dir,  f"elev_{int(self.args.sun_el)}_azim_{int(self.args.sun_az)}")
            if not os.path.isdir(map_store_dir):
                os.makedirs(map_store_dir)
            # Render:
            blend_utils.render_tiles_to_file(map_scene, map_cam, map_store_dir, img_id=idx, tiles_x=map_tiles_x, tiles_y=map_tiles_y)
            # Save depth
            data_utils.save_depth_tiles(map_scene, map_cam, map_store_dir, id=idx, tiles_x=map_tiles_x, tiles_y=map_tiles_y, save_png=True)   
            # Save data
            data_utils.save_pose_data(map_cam, self.env_config["ortho_cam_altitude"], map_store_dir, id=idx)
            
            # Load and data
            map_depth = data_utils.load_depth(f"{map_store_dir}/depth/{map_cam.name}" + "_%04i" % idx + "_depth")
            map_gray  = data_utils.paste_map_tiles(tiles_dir=f"{map_store_dir}/images", overall_width=map_width, overall_height=map_height, n_tiles_x=map_tiles_x, n_tiles_y=map_tiles_y)        
        else:
            map_dir = self.args.map_dir
            # Load cam data
            with open(os.path.join(map_dir, "cam_data.json")) as f:
                map_cam_data = json.load(f)
            map_cam_name = map_cam_data['name']
            map_width, map_height = map_cam_data['img_width'], map_cam_data['img_height']
            # Load params data
            with open(os.path.join(map_dir, "params.json" )) as f:
                map_params = json.load(f)
            map_tiles_x = map_params['tiles_x']
            map_tiles_y = map_params['tiles_y']
            # Load image and depth
            map_store_dir = os.path.join(map_dir, f"elev_{int(self.args.sun_el)}_azim_{int(self.args.sun_az)}")
            map_depth = data_utils.load_depth(f"{map_store_dir}/depth/{map_cam_name}" + "_%04i" % idx + "_depth")
            map_gray  = data_utils.paste_map_tiles(tiles_dir=f"{map_store_dir}/images", overall_width=map_width, overall_height=map_height, n_tiles_x=map_tiles_x, n_tiles_y=map_tiles_y)        
        print(f"\nMap depth shape: {map_depth.shape} ")
        print(f"\nMap gray shape: {map_gray.shape} ")

        ##### Print data
        # Poses
        print(f"--- Cam 0 pose---")
        print(f"Location in Blender world reference frame: (x, y, z) = ({cam0_loc[0]} , {cam0_loc[1]}, {cam0_loc[2]})")
        print(f"Altitude: {self.args.cam0_altitude} m") 
        print(f"Euler angles: (Yaw, Pitch, Roll) = {(self.args.cam0_yaw, self.args.cam0_pitch, self.args.cam0_roll)} deg")

        print(f"--- Cam 1 pose---")
        print(f"Location in Blender world reference frame: (x, y, z) = ({cam1_loc[0]} , {cam1_loc[1]}, {cam1_loc[2]})")
        print(f"Altitude: {self.args.cam1_altitude} m") 
        print(f"Euler angles: (Yaw, Pitch, Roll) = {(self.args.cam1_yaw, self.args.cam1_pitch, self.args.cam1_roll)} deg")

        print(f"------------ Map camera pose ------------")
        print(f"Location in Blender world reference frame: (x, y, z) = ({map_cam.location[0]} , {map_cam.location[1]} , {map_cam.location[2]})")
        print(f"Euler angles: (Yaw, Pitch, Roll) = (0, 0, 0) deg")
        print(f"------------ Save Metadata ------------")

        print(f"\n------------ Sun Angle ------------")
        print(f"(Azimuth, Elevation) = ({self.args.sun_az} , {self.args.sun_el} ) deg")


        ##### Project points from Map to Cam0 and viceversa
        get_cam2cam_correspondences(map_scene, map_cam, map_gray, map_depth, cam0_scene, cam0, cam0_gray, cam0_depth, bound=0, px_sampling_step=100, plot_on_dmap=True, dest_dir=map_dir)
        

        ##### Show depth maps
        # map_depth_img = normalize_depth(load_depth(map_scene), map_cam)
        # cam0_depth_img = normalize_depth(load_depth(cam0_scene), cam0)
        # show_image_pair(map_depth_img, cam0_depth_img, title1='Map', title2='Cam0')
        # cam0_gray = cv2.imread(f"tmp/{cam0_scene.name}.png")
        # show_image_pair(cam0_gray, cam0_depth_img, title1='Cam0 Image', title2='Cam0 Depth')

        # Show map:
        fig, ax = plt.subplots()
        ax.imshow(map_gray, aspect='equal')
        ax.set_title("Map")
        ax.axis("off")
        plt.tight_layout()
        plt.show(block=True)

        # # Clear the temporary directory
        # for filename in os.listdir("./tmp"):
        #     file_path = os.path.join("./tmp", filename)
        #     try:
        #         if os.path.isfile(file_path) or os.path.islink(file_path):
        #             os.unlink(file_path)
        #         elif os.path.isdir(file_path):
        #             shutil.rmtree(file_path)
        #     except Exception as e:
        #         print(f'Failed to delete {file_path}. Reason: {e}')


    def run_dataset(self):
        print("\n**************** Running dataset generation ****************************")

        # Set attributes
        light = self.light
        map_cam = self.map_cam
        map_scene = self.map_scene
        mapcam_config_manager = self.mapcam_config_manager
        query_cam = self.cam0
        query_scene = self.cam0_scene
        querycam_config_manager = self.cam0_config_manager

        
        # Sample Sun elevation and azimuth (in deg)
        sun_ELs = np.arange(self.args.sun_EL_range[0], self.args.sun_EL_range[1] + self.args.sun_EL_step, self.args.sun_EL_step)
        sun_AZs = np.arange(self.args.sun_AZ_range[0], self.args.sun_AZ_range[1] + self.args.sun_AZ_step, self.args.sun_AZ_step)


        ##### RENDER QUERIES #########
        if self.args.render_queries:
            ## QUERY INPUT METADATA (independent of Sun and Pose)
            #   Set repos
            if not os.path.isdir(self.args.query_dest):
                os.makedirs(self.args.query_dest)
            if not os.path.isdir(self.args.query_dest + f"/configs/"):
                os.makedirs(self.args.query_dest + f"/configs/")
            #   Save cam data
            data_utils.save_cam_data(query_cam, query_scene, self.args.query_dest)
            #   Save config files
            self.env_config_manager.save_config(self.args.query_dest + f"/configs/{self.args.main_yaml}")
            querycam_config_manager.save_config(self.args.query_dest + f"/configs/querycam.yaml")
            #   Save config files from Blender (for a sanity check)
            data_utils.save_config_from_blender(query_scene, light, self.args.query_dest + f"/configs/")
            
            #   Save params
            data_utils.save_query_dataset_params(self.args)       

            # Compute margin based on query cam and max altitude
            query_img_width, query_img_height = query_scene.render.resolution_x, query_scene.render.resolution_y
            query_img_aspect_x, query_img_aspect_y = query_scene.render.pixel_aspect_x, query_scene.render.pixel_aspect_y
            query_img_aspect_ratio = query_img_aspect_x / query_img_aspect_y
            query_focal_length = query_cam.data.lens
            query_sensor_width = query_cam.data.sensor_width
            swath_x = self.args.altitude_range[1]*(query_sensor_width / query_focal_length)
            swath_y = swath_x*query_img_aspect_ratio*(query_img_height / query_img_width)
            width_margin = 0.5*swath_x
            height_margin = 0.5*swath_y
            print(f"Margins: (w_margin, h_margin)={(width_margin,height_margin)}")

            # Extract terrain pose and size from the orthographic map used as texture image (reference)
            ref_map_ortho_size = [self.map_ortho_width, self.map_ortho_height]
            map_manager = OrthoImageManager(img_path=self.env_config['Terrain']['low_res_ortho'], img_ortho_size=ref_map_ortho_size)
            ref_map_img = map_manager.img
            ref_map_size = map_manager.img_size
            ref_map_px_res = map_manager.px_res # m/px
            terrain_ortho_size = map_manager.terrain_ortho_size
            R_bt = map_manager.rot_matrix
            t_bt = map_manager.center_blend


            if self.args.sample_area:

                # Sampling camera locations on terrain, in Blender frame
                if self.args.fixed_obs:
                    query_locs_x = np.linspace(self.args.xrange[0] + width_margin,  self.args.xrange[1] - width_margin, self.args.samples)
                    query_locs_y = np.linspace(self.args.yrange[0] + height_margin, self.args.yrange[1] - height_margin, self.args.samples)
                    altitudes    =  np.linspace(self.args.altitude_range[0], self.args.altitude_range[1], self.args.samples)
                else:
                    query_locs_x = np.random.uniform(low=self.args.xrange[0] + width_margin,  high=self.args.xrange[1] - width_margin, size=self.args.samples)
                    query_locs_y = np.random.uniform(low=self.args.yrange[0] + height_margin, high=self.args.yrange[1] - height_margin, size=self.args.samples)
                    altitudes    = np.random.uniform(low=self.args.altitude_range[0], high=self.args.altitude_range[1], size=self.args.samples)
                query_locs_b = np.zeros((self.args.samples,2))
                for i in range(self.args.samples):
                    query_locs_b[i,:] = np.array([query_locs_x[i], query_locs_y[i]])
                
            else:

                # Sampling camera locations on terrain, in terrain frame
                yrange_m = (-terrain_ortho_size[1]/2 + height_margin, terrain_ortho_size[1]/2 - height_margin)
                xrange_m = (-terrain_ortho_size[0]/2 + width_margin, terrain_ortho_size[0]/2 - width_margin)
                if self.args.fixed_obs:
                    query_locs_x = np.linspace(xrange_m[0], xrange_m[1], self.args.samples)
                    query_locs_y = np.linspace(yrange_m[0], yrange_m[1], self.args.samples)
                    altitudes =  np.linspace(self.args.altitude_range[0], self.args.altitude_range[1], self.args.samples)
                else:
                    query_locs_x = np.random.uniform(low=xrange_m[0], high=xrange_m[1], size=self.args.samples)
                    query_locs_y = np.random.uniform(low=yrange_m[0], high=yrange_m[1], size=self.args.samples)
                    altitudes =  np.random.uniform(low=self.args.altitude_range[0], high=self.args.altitude_range[1], size=self.args.samples)
                print(f"altitudes: {altitudes}")
                #   Convert to the Blender world frame
                query_locs_b = np.zeros((self.args.samples,2))
                for i in range(self.args.samples):
                    query_loc_m = np.array([query_locs_x[i], query_locs_y[i]])
                    query_loc_b = np.dot(R_bt, query_loc_m) + t_bt
                    query_locs_b[i,:] = query_loc_b

            
            #   Display the result using Matplotlib
            fig_sampled, ax = plt.subplots(figsize=(4,6))
            ax.imshow(ref_map_img)
            for i in range(self.args.samples):
                point_blender = query_locs_b[i,:]
                point_px = blender2img(point_blender, ref_map_size, ref_map_px_res)
                ax.plot(point_px[0], point_px[1], marker="o", color='r', markersize='2')
            ax.set_title("Sampled location")
            ax.axis('off')  # Hide the axis
            # plt.show()
            # Save sampled locations img
            fig_sampled.savefig(f"{self.args.query_dest}/sampled_locations.png", format='png', dpi=300)
            print(f"sampled_locations.png saved at {self.args.query_dest}")
            # plt.show()

            # Sample pose perturbations (in deg)
            pitches = np.random.uniform(low=-self.args.max_pitch, high=self.args.max_pitch, size=self.args.samples) 
            rolls = np.random.uniform(low=-self.args.max_roll, high=self.args.max_roll, size=self.args.samples)
            yaws = np.random.uniform(low=-self.args.max_yaw, high=self.args.max_yaw, size=self.args.samples)
            print(f"pitches: {pitches}")
            print(f"rolls: {rolls}")
            print(f"yaws: {yaws}")

            
            ## For every sampled pose (loc, altitude, cam rot) render images with all elev and azimuth of interest   
            for i in range(len(sun_ELs)):
                    for j in range(len(sun_AZs)):

                        # Set query output repo
                        query_store_dir = self.args.query_dest + "/elev_" + str(int(sun_ELs[i])) + "_azim_" + str(int(sun_AZs[j]))
                        if not os.path.isdir(query_store_dir):
                            os.makedirs(query_store_dir)

                        # Set Sun position
                        light.rotation_euler = get_rot_tuple((0, 90. - sun_ELs[i], sun_AZs[j]))

                                
                        for k in range(self.args.samples):
                            print(f"sample {k}")
                            # Set Query pose
                            blend_utils.set_camera_above_terrain(query_cam, self.terrain_mesh, query_locs_b[k,0], query_locs_b[k,1], altitudes[k]) 
                            print(f"Set query above terrain")
                            query_rot = [pitches[k], rolls[k], yaws[k]] # X, Y, Z euler angles in B frame (deg) ([0,0,0] points downwards)
                            query_cam.rotation_euler = get_rot_tuple(query_rot) # (radians)
                            print(f"Set query pose complete")
                            # Render:
                            blend_utils.render_to_file(query_scene, query_cam, query_store_dir, img_id=k)
                            print(f"Render complete")
                            # Save depth
                            data_utils.save_depth(query_scene, query_cam, query_store_dir, id=k, save_png=True)

                            # Save metadata
                            data_utils.save_pose_data(query_cam, altitudes[k], query_store_dir, id=k)  
        
        
        ##### RENDER MAPS #########
        if self.args.render_map:
            print(f"RENDER MAPS")
            ## MAP INPUT METADATA (independent of Sun and Pose)
            #   Set repos
            if not os.path.isdir(self.args.map_dest):
                os.makedirs(self.args.map_dest)
            if not os.path.isdir(self.args.map_dest + f"/configs/"):
                os.makedirs(self.args.map_dest + f"/configs/")
            #   Save cam data
            data_utils.save_cam_data(map_cam, map_scene, self.args.map_dest)
            #   Save config files
            self.env_config_manager.save_config(self.args.map_dest + f"/configs/{self.args.main_yaml}")
            mapcam_config_manager.save_config(self.args.map_dest + f"/configs/mapcam.yaml")
            #   Save config files from Blender (for a sanity check)
            data_utils.save_config_from_blender(map_scene, light, self.args.map_dest +f"/configs/")
            
            #   Save light params
            data_utils.save_map_dataset_params(self.args)   

            # Render image with all elev and azimuth of interest
            for i in range(len(sun_ELs)):
                    for j in range(len(sun_AZs)):
                            
                        # Set map output repo
                        map_store_dir = self.args.map_dest + "/elev_" + str(int(sun_ELs[i])) + "_azim_" + str(int(sun_AZs[j]))
                        if not os.path.isdir(map_store_dir):
                            os.makedirs(map_store_dir)

                        # Set Sun position
                        light.rotation_euler = get_rot_tuple((0, 90. - sun_ELs[i], sun_AZs[j]))

                        # Set Map pose
                        blend_utils.set_camera_above_terrain(map_cam, self.terrain_mesh,  self.env_config["ortho_cam_x"], self.env_config["ortho_cam_y"], self.env_config["ortho_cam_altitude"])  
                        map_cam.rotation_euler = (0,0,0)

                        # Render:
                        # render_to_file(map_scene, map_cam, map_store_dir + f"/images", img_id=0)
                        # TODO: include the possibility to not render depth
                        blend_utils.render_tiles_to_file(map_scene, map_cam, map_store_dir, img_id=0, tiles_x=self.args.tiles_x, tiles_y=self.args.tiles_y)
                        
                        # Save metadata
                        data_utils.save_pose_data(map_cam, self.env_config["ortho_cam_altitude"], map_store_dir, id=0)

            # Render map depth (same for any Sun (AZ,EL))
            exr_files_dir = os.path.join(map_store_dir, "exr_files")
            if self.args.save_map_depth:
                map_store_dir = os.path.join(self.args.map_dest, "map_depth")
                data_utils.save_depth_tiles(map_scene, map_cam, map_store_dir, id=0, tiles_x=self.args.tiles_x, tiles_y=self.args.tiles_y, save_png=True)
                print(f"Map depth rendered at {map_store_dir}/depth")
                shutil.move(exr_files_dir, os.path.join(map_store_dir, "map_depth"))
            else:
                shutil.rmtree(exr_files_dir)

