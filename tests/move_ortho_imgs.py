import os
import shutil

# Define the ranges and steps
sun_EL_range = (30, 90)  # Replace with your actual range
sun_EL_step = 10        # Replace with your actual step
sun_AZ_range = (0, 315) # Replace with your actual range
sun_AZ_step = 45       # Replace with your actual step

# Generate the list of elevations and azimuths
elevations = list(range(sun_EL_range[0], sun_EL_range[1] + sun_EL_step, sun_EL_step))
azimuths = list(range(sun_AZ_range[0], sun_AZ_range[1] + sun_AZ_step, sun_AZ_step))

# Iterate over all combinations of elevations and azimuths
for elevation in elevations:
    for azimuth in azimuths:
        # Construct source and destination paths
        source_path = f"dataset/samples/queries/altitudes/elev_{elevation}_azim_{azimuth}/images/OrthoCam_0000.png"
        dest_dir = f"dataset/samples/maps/elev_{elevation}_azim_{azimuth}/images"
        dest_path = f"{dest_dir}/OrthoCam_0000.png"
        
        # Create destination directory if it doesn't exist
        os.makedirs(dest_dir, exist_ok=True)
        
        # Move the file
        if os.path.exists(source_path):
            shutil.move(source_path, dest_path)
            print(f"Moved {source_path} to {dest_path}")
        else:
            print(f"Source file {source_path} does not exist.")

print("Files moved successfully.")
