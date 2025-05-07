import os
import glob
from PIL import Image
import matplotlib.pyplot as plt

# Define the parameters

sun_AZ_range = [0, 315]  # Define your actual azimuth range (deg)
sun_AZ_step = 45         # Define your actual step (deg)

sun_EL_range = [20, 90]  # Define your actual elevation range (deg)
sun_EL_step = 10         # Define your actual step (deg)

# Generate the list of azimuth angles
sun_AZs = list(range(sun_AZ_range[0], sun_AZ_range[1] + sun_AZ_step, sun_AZ_step))
sun_ELs = list(range(sun_EL_range[0], sun_EL_range[1] + sun_EL_step, sun_EL_step))

# Set an output directory
map_out_dir = "./gifs/maps"
if not os.path.isdir(map_out_dir):
    os.makedirs(map_out_dir)
query_out_dir = "./gifs/queries"
if not os.path.isdir(query_out_dir):
    os.makedirs(query_out_dir)

# Set a temporary directory
tmp_dir = "./tmp_gif"
if not os.path.isdir(tmp_dir):
    os.makedirs(tmp_dir)

# GENERATE GIF WITH VARYING AZIMUTH
for EL in sun_ELs:
    # Path to images
    az_image_paths = [f"dataset/samples/maps/elev_{EL}_azim_{azim}/images/OrthoCam_0000.png" for azim in sun_AZs]
    # List to store images for GIF
    gif_images = []
    # Iterate over image paths, set titles, and prepare for GIF
    for i, image_path in enumerate(az_image_paths):
        # Open image
        img = Image.open(image_path)
        
        # Create a plot to add title
        fig, ax = plt.subplots(figsize=(4,6), dpi=300)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"Sun EL = {EL} deg, AZ = {sun_AZs[i]} deg")
        
        # Save the plot with title as a new image
        temp_image_path = f"{tmp_dir}/temp_image_{i}.png"
        plt.savefig(temp_image_path)
        plt.close(fig)
        
        # Open the new image and append to the list
        gif_images.append(Image.open(temp_image_path))

    # Save as GIF
    gif_images[0].save(f"{map_out_dir}/elev_{EL}_azimVar.gif", save_all=True, append_images=gif_images[1:], duration=500, loop=0)

    # Clean up temporary images
    for i in range(len(sun_AZs)):
        os.remove(f"{tmp_dir}/temp_image_{i}.png")

    print(f"'elev_{EL}_azimVar.gif' GIF created successfully.")


# GENERATE GIF WITH VARYING ELEVATION
for AZ in sun_AZs:
    # Path to images
    el_image_paths = [f"dataset/samples/maps/elev_{elev}_azim_{AZ}/images/OrthoCam_0000.png" for elev in sun_ELs]

    # List to store images for GIF
    gif_images = []
    # Iterate over image paths, set titles, and prepare for GIF
    for i, image_path in enumerate(el_image_paths):
        # Open image
        img = Image.open(image_path)
        
        # Create a plot to add title
        fig, ax = plt.subplots(figsize=(4,6), dpi=300)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"Sun EL = {sun_ELs[i]} deg, AZ = {AZ} deg")
        
        # Save the plot with title as a new image
        temp_image_path = f"{tmp_dir}/temp_image_{i}.png"
        plt.savefig(temp_image_path)
        plt.close(fig)
        
        # Open the new image and append to the list
        gif_images.append(Image.open(temp_image_path))

    # Save as GIF
    gif_images[0].save(f"{map_out_dir}/azim_{AZ}_elevVar.gif", save_all=True, append_images=gif_images[1:], duration=500, loop=0)

    # Clean up temporary images
    for i in range(len(sun_ELs)):
        os.remove(f"{tmp_dir}/temp_image_{i}.png")

    print(f"'azim_{AZ}_elevVar.gif' GIF created successfully.")

