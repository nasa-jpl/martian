

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


import os
import sys
# Assuming this is ran in the root of the europa_sim repo. 
# Add current directory in path
BASE_PATH = os.getcwd() + "/"
sys.path.append(BASE_PATH)

import argparse
from configs import Config
from classes.blender_runner import BlenderRunner
from classes.ortho_image_manager import OrthoImageManager
# os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

def validate_common_args(args):
    # args = parser.parse_args(argv)
    
    if args.blend_file is None and not args.generate_blend_file:
        sys.exit("Error: At least one of --blend_file or --generate_blend_file must be provided.")

    if args.generate_blend_file:
        if args.dtm_resolution is None:
            sys.exit("Error: --generate_blend_file requires --dtm_resolution to be provided.")
        else:
            print(f"Blender file generation enabled with DTM resolution: {args.dtm_resolution}")

    if args.save_blend:
        if args.blend_dest is None or args.blend_filename is None:
            sys.exit("Error: --save_blend requires --blend_dest and --blend_filename to be provided.")
        else:
            print(f"Blender file destination: {args.blend_dest}")
            print(f"Blender file name: {args.blend_filename}")

# def validate_demo_args(parser, argv):
#     args = parser.parse_args(argv)

#     # Check if any mismatch exists: all or none must be provided
#     if (args.map_img_path is None) != (args.map_depth_path is None) or (args.map_img_path is None) != (args.map_cam_data is None):
#         parser.error("All three arguments --map_img_path, --map_depth_path, and --map_cam_data must be provided together, or none of them.")



def validate_dataset_args(parser, argv):
    args = parser.parse_args(argv)
    if not args.render_queries and not args.render_map:
        sys.exit("Error: At least one of --render_queries or --render_map must be provided.")
    
    if args.render_queries:
        required_query_args = ['query_dest', 'samples', 'altitude_range', 'max_yaw', 'max_pitch', 'max_roll']
        for arg in required_query_args:
            if getattr(args, arg) is None:
                sys.exit(f"Error: --render_queries requires --{arg.replace('_', '-')} to be provided.")
    
    if args.render_map:
        required_map_args = ['map_dest', 'map_px_res']
        for arg in required_map_args:
            if getattr(args, arg) is None:
                sys.exit(f"Error: --render_map requires --{arg.replace('_', '-')} to be provided.")


def parse_common_arguments():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--mode', choices=['demo', 'dataset'], default="demo", help='Modes of operation')
    parser.add_argument('--main_yaml', type=str, default='jezero.yaml', help="Path to the main YAML configuration file")
    parser.add_argument('--keep_blender_running', default=False, action='store_true', help="If enabled, keeps blender process running after rendering.")
    
    blender_file_group = parser.add_mutually_exclusive_group()
    blender_file_group.add_argument('--blend_file', type=str, default="hirise_input/jezero_crater/blend_files/jezero_40resDTM_C01ortho", help="file path of the blender file to load")
    # blender_file_group.add_argument('--blend_file', default=None, help="file path of the blender file to load")
    blender_file_group.add_argument('--generate_blend_file', default=False, action='store_true', help="If enabled, generate a blender file from the info in the main yaml file.")
    

    parser.add_argument('--dtm_resolution', type=float, default=10., help="Percentage scale for terrain model resolution. 100\% loads the "
                                                                            "model at full resolution (i.e. one vertex for each post in the "
                                                                            "original terrain model) and is *MEMORY INTENSIVE*. Downsampling "
                                                                            "uses Nearest Neighbors. The downsampling algorithm may need to "
                                                                            "alter the resolution you specify here to ensure it results in a "
                                                                            "whole number of vertices. If it needs to alter the value you "
                                                                            "specify, you are guaranteed that it will shrink it (i.e. "
                                                                            "decrease the DTM resolution.")
    parser.add_argument('--save_blend', default=False, action='store_true', help="If enabled, it save the blender file")
    parser.add_argument('--blend_dest', type=str, default="hirise_input/jezero_crater/blend_files", help="Destination folder of the blender file.")    
    parser.add_argument('--blend_filename', type=str, default="jezero_10resDTM", help="Filename of the blender file.")
    parser.add_argument('--save_blend_scenes', default=False, action='store_true', help="If enabled, it save the scenes in a blender file")   
    return parser

def parse_demo_arguments(common_parser):
    parser = argparse.ArgumentParser(description="Blender demo mode", parents=[common_parser])
    parser.add_argument('--demo_name', type=str, default="demo", help='Name of the demo')
    parser.add_argument('--dest_dir', type=str, default="./", help='Name of the demo')
    parser.add_argument('--sun_az', type=float, default=0., help='Sun Azimuth angle (deg)')
    parser.add_argument('--sun_el', type=float, default=90., help='Sun Elevation angle (deg)')
    parser.add_argument('--map_px_res', type=float, default=10, help='Pixel resolution of rendered map. Default: 10 m / pixel.')
    parser.add_argument('--map_dir', type=str, default=None, help='Path to a map storage directory generated with mars-sim, including map tile images, depth and configuration informaiton. Default:None. If provided, the demo will use that without rendering a new map.') 
    parser.add_argument('--bound', type=float, default=0., help='Defines the margin (in pixels) to exclude from the edges of the images during points projections.')
    
    cam0_parser = parser.add_argument_group('Camera 0', 'Group of arguments for camera 0')
    cam0_parser.add_argument('--cam0_altitude', type=float, default=700., help='Altitude of the persp camera (m)')
    cam0_parser.add_argument('--cam0_loc_x', type=float, default=-1930., help='X Location of the persp cam in Blender reference frame (m)')
    cam0_parser.add_argument('--cam0_loc_y', type=float, default=1800., help='Y Location of the persp cam in Blender reference frame (m)')
    cam0_parser.add_argument('--cam0_yaw', type=float, default=0., help='Yaw (deg) - Angle around Z axis (pointing upwards)')
    cam0_parser.add_argument('--cam0_pitch', type=float, default=10., help='Pitch (deg) - Angle around X axis (pointing rightward w.r.t. the direction of camera motion)')
    cam0_parser.add_argument('--cam0_roll', type=float, default=20., help='Roll (deg) - Angle around Y axis (pointing towards the direction of camera motion)')
    cam0_parser.set_defaults()

    cam1_parser = parser.add_argument_group('Camera 1', 'Group of arguments for camera 1')
    cam1_parser.add_argument('--cam1_altitude', type=float, default=700., help='Altitude of the persp camera (m)')
    cam1_parser.add_argument('--cam1_loc_x', type=float, default=-1950., help='X Location of the persp cam in Blender reference frame (m) - for negative values be sure to provide the value in quote and with space: e.g. " -1678"')
    cam1_parser.add_argument('--cam1_loc_y', type=float, default=1800., help='Y Location of the persp cam in Blender reference frame (m)')
    cam1_parser.add_argument('--cam1_yaw', type=float, default=0., help='Yaw (deg) - Angle around Z axis (pointing downwards)')
    cam1_parser.add_argument('--cam1_pitch', type=float, default=0., help='Pitch (deg) - Angle around Y axis (pointing rightward w.r.t. the direction of motion)')
    cam1_parser.add_argument('--cam1_roll', type=float, default=0., help='Roll (deg) - Angle around X axis (pointing towards the direction of motion)')
    cam1_parser.set_defaults()

    return parser

def parse_dataset_arguments(common_parser):
    parser = argparse.ArgumentParser(description="Blender dataset mode", parents=[common_parser])
    
    # Common dataset args
    parser.add_argument('--sun_EL_range', type=float, nargs=2, default=[20,90], help="Sun Elevation range (deg)", required=False)
    parser.add_argument('--sun_AZ_range', type=float, nargs=2, default=[0,315], help="Sun Azimuth range (deg)", required=False)
    parser.add_argument('--sun_EL_step', type=float, default=10, help="Sun Elevation step (deg)", required=False)
    parser.add_argument('--sun_AZ_step', type=float, default=45, help="Sun Azimuth step (deg)", required=False)

    # Query args
    queries_group = parser.add_argument_group("Arguments for queries renders")
    queries_group.add_argument("--render_queries", default=False, action='store_true', help="If enabled, render queries at the specified Sun EL, AZ.")
    queries_group.add_argument("--query_dest", default='./dataset/samples/queries/altitudes', help="Dataset with query images", required=False)
    queries_group.add_argument('--samples', type=int, default=10, help="Number of query samples", required=False)  
    queries_group.add_argument('--altitude_range', type=float, nargs=2, default=[50,150], help="Altitude range (m)", required=False)
    queries_group.add_argument('--max_yaw', type=float, default=0., help="Max absolute value of yaw angle (deg)", required=False)
    queries_group.add_argument('--max_pitch', type=float, default=0., help="Max absolute value of pitch angle (deg)", required=False)
    queries_group.add_argument('--max_roll', type=float, default=0., help="Max absolute value of roll angle (deg)", required=False)
    queries_group.add_argument("--sample_area", default=False, action='store_true', help="If enabled, queries are sample from the square area with specified coorner coordinates")
    queries_group.add_argument("--xrange", default=[-2500., 500.], help="x-coordinates range in Blender frame [m]")
    queries_group.add_argument("--yrange", default=[ 1000., 6000.], help="y-coordinates range in Blender frame [m]")
    queries_group.add_argument("--fixed_obs", default=False, action='store_true', help="If enabled, queries are linearly uniformly distirbuted in the ranges above (if disabled, the sampling is random)")
    
    # Map args
    map_group = parser.add_argument_group("Arguments for map renders")
    map_group.add_argument("--render_map", default=False, action='store_true', help="If enabled, render also different tiles map at the specified Sun EL, AZ.")
    map_group.add_argument("--save_map_depth", default=False, action='store_true', help="If enabled, save depth data for the rendered map tiles.")
    map_group.add_argument('--map_dest', type=str, default='./dataset/samples/maps/', required=False)
    map_group.add_argument('--map_px_res', type=float, default=10, help='Pixel resolution of rendered map. Default: 10 m / pixel.')
    map_group.add_argument('--tiles_x', type=int, default=8, help='Number of tiles to render along x')
    map_group.add_argument('--tiles_y', type=int, default=16, help='Number of tiles to render along y')
    # TODO:
    # add possibility to render a specific tile (i,j) with i in range(tile_x), j in range (tile_y)
    
    return parser


def main():

    print("sys.argv:", sys.argv)
    if '--' not in sys.argv:
        argv = []
    else:
        argv = sys.argv[sys.argv.index('--') + 1:]
    print("argv passed to parsers:", argv)

    # Define the common arguments
    common_parser = parse_common_arguments()
    initial_args = common_parser.parse_known_args(argv)[0]
    validate_common_args(initial_args)

    # Based on the mode, parse the full arguments with the appropriate parser
    print(f"MODE: {initial_args.mode}")
    if initial_args.mode == 'demo':
        parser = parse_demo_arguments(common_parser)
        # validate_demo_args(parser, argv) # Check that both map_img_path and map_depth path arguments are provided whenever one of them is provided
    elif initial_args.mode == 'dataset':
        parser = parse_dataset_arguments(common_parser)
        validate_dataset_args(parser, argv) # Check that at least one betwenn --render_queries and --render_map is provided
    
    # Parse the full set of arguments
   
    args = parser.parse_args(argv)
    print("Parsed arguments:", args)
    

    # Run the BlenderRunner with the parsed arguments
    runner = BlenderRunner(args)
    runner.run()

if __name__ == '__main__':
    main()