import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
from fpdf import FPDF

# Set a temporary directories
tmp_dir_gif = "./tmp_gif"
if not os.path.isdir(tmp_dir_gif):
    os.makedirs(tmp_dir_gif)


# Set a temporary directories
tmp_dir_pdf = "./tmp_pdf"
if not os.path.isdir(tmp_dir_pdf):
    os.makedirs(tmp_dir_pdf)



def create_query_gif_with_varying_azimuth(EL, data_path, sample_id, gif_dest_dir):
    # GENERATE GIF WITH VARYING AZIMUTH
    # Path to images
    az_image_paths = [f"{data_path}/elev_{EL}_azim_{azim}/images/PerspCam0_000{sample_id}.png" for azim in sun_AZs]
    # List to store images for GIF
    gif_images = []
    # Iterate over image paths, set titles, and prepare for GIF
    for i, image_path in enumerate(az_image_paths):
        # Open image
        img = Image.open(image_path)
        
        # Create a plot to add title
        fig, ax = plt.subplots(figsize=(12,10), dpi=300)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"Sun EL = {EL} deg, AZ = {sun_AZs[i]} deg", fontsize=20)
        
        # Save the plot with title as a new image
        temp_image_path = f"{tmp_dir_gif}/temp_image_{i}.png"
        plt.savefig(temp_image_path)
        plt.close(fig)
        
        # Open the new image and append to the list
        gif_images.append(Image.open(temp_image_path))

    # Save as GIF
    gif_images[0].save(f"{gif_dest_dir}/id{sample_id}_elev_{EL}_azimVar.gif", save_all=True, append_images=gif_images[1:], duration=500, loop=0)

    # Clean up temporary images
    for i in range(len(sun_AZs)):
        os.remove(f"{tmp_dir_gif}/temp_image_{i}.png")

    print(f"'id{sample_id}_elev_{EL}_azimVar.gif' GIF created successfully.")

def create_map_gif_with_varying_azimuth(EL, data_path, gif_dest_dir):
    # GENERATE GIF WITH VARYING AZIMUTH
    # Path to images
    az_image_paths = [f"{data_path}/elev_{EL}_azim_{azim}/images/OrthoCam_0000.png" for azim in sun_AZs]
    # List to store images for GIF
    gif_images = []
    # Iterate over image paths, set titles, and prepare for GIF
    for i, image_path in enumerate(az_image_paths):
        # Open image
        img = Image.open(image_path)
        
        # Create a plot to add title
        fig, ax = plt.subplots(figsize=(12,10), dpi=300)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"Sun EL = {EL} deg, AZ = {sun_AZs[i]} deg", fontsize=20)
        
        # Save the plot with title as a new image
        temp_image_path = f"{tmp_dir_gif}/temp_image_{i}.png"
        plt.savefig(temp_image_path)
        plt.close(fig)
        
        # Open the new image and append to the list
        gif_images.append(Image.open(temp_image_path))

    # Save as GIF
    gif_images[0].save(f"{gif_dest_dir}/elev_{EL}_azimVar.gif", save_all=True, append_images=gif_images[1:], duration=500, loop=0)

    # Clean up temporary images
    for i in range(len(sun_AZs)):
        os.remove(f"{tmp_dir_gif}/temp_image_{i}.png")

    print(f"'elev_{EL}_azimVar.gif' GIF created successfully.")

def create_query_gif_with_varying_elevation(AZ, data_path, sample_id, gif_dest_dir):
    # Path to images
    el_image_paths = [f"{data_path}/elev_{elev}_azim_{AZ}/images/PerspCam0_000{sample_id}.png" for elev in sun_ELs]
    # List to store images for GIF
    gif_images = []
    # Iterate over image paths, set titles, and prepare for GIF
    for i, image_path in enumerate(el_image_paths):
        # Open image
        img = Image.open(image_path)
        
        # Create a plot to add title
        fig, ax = plt.subplots(figsize=(12,10), dpi=300)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"Sun EL = {sun_ELs[i]} deg, AZ = {AZ} deg", fontsize=20)
        
        # Save the plot with title as a new image
        temp_image_path = f"{tmp_dir_gif}/temp_image_{i}.png"
        plt.savefig(temp_image_path)
        plt.close(fig)
        
        # Open the new image and append to the list
        gif_images.append(Image.open(temp_image_path))

    # Save as GIF
    gif_images[0].save(f"{gif_dest_dir}/id{sample_id}_azim_{AZ}_elevVar.gif", save_all=True, append_images=gif_images[1:], duration=500, loop=0)

    # Clean up temporary images
    for i in range(len(sun_ELs)):
        os.remove(f"{tmp_dir_gif}/temp_image_{i}.png")

    print(f"'id{sample_id}_azim_{AZ}_elevVar.gif' GIF created successfully.")

def create_map_gif_with_varying_elevation(AZ, data_path,gif_dest_dir):
    # Path to images
    el_image_paths = [f"{data_path}/elev_{elev}_azim_{AZ}/images/OrthoCam_0000.png" for elev in sun_ELs]
    # List to store images for GIF
    gif_images = []
    # Iterate over image paths, set titles, and prepare for GIF
    for i, image_path in enumerate(el_image_paths):
        # Open image
        img = Image.open(image_path)
        
        # Create a plot to add title
        fig, ax = plt.subplots(figsize=(12,10), dpi=300)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"Sun EL = {sun_ELs[i]} deg, AZ = {AZ} deg", fontsize=20)
        
        # Save the plot with title as a new image
        temp_image_path = f"{tmp_dir_gif}/temp_image_{i}.png"
        plt.savefig(temp_image_path)
        plt.close(fig)
        
        # Open the new image and append to the list
        gif_images.append(Image.open(temp_image_path))

    # Save as GIF
    gif_images[0].save(f"{gif_dest_dir}/azim_{AZ}_elevVar.gif", save_all=True, append_images=gif_images[1:], duration=500, loop=0)

    # Clean up temporary images
    for i in range(len(sun_ELs)):
        os.remove(f"{tmp_dir_gif}/temp_image_{i}.png")

    print(f"'azim_{AZ}_elevVar.gif' GIF created successfully.")

# def create_pdf_with_varying_azimuth(EL, data_path, sample_id, pdf_dest_dir):
#     # Function to create a PDF with varying azimuth
#     pdf = FPDF()
#     for i, azim in enumerate(sun_AZs):
#         image_path = f"{data_path}/altitudes/elev_{EL}_azim_{azim}/images/PerspCam0_000{sample_id}.png"
#         # temp_image_path = f"{tmp_dir_pdf}/temp_image_{i}.png"
#         if os.path.exists(image_path):
#             # convert_to_8bit(image_path, temp_image_path)
#             pdf.add_page()
#             pdf.set_font("Arial", size=12)
#             pdf.cell(200, 10, txt=f"Sun EL = {EL} deg, AZ = {azim} deg", ln=True, align='C')
#             pdf.image(image_path, x=10, y=20, w=180)  # Adjust the position and size as needed

#     pdf.output(f"{pdf_dest_dir}/id{sample_id}_elev_{EL}_azimVar.pdf")
#     print(f"'id{sample_id}_elev_{EL}_azimVar.pdf' PDF created successfully.")

# def create_pdf_with_varying_elevation(AZ, data_path, sample_id, pdf_dest_dir):
#     # Function to create a PDF with varying elevation
#     pdf = FPDF()
#     for i, elev in enumerate(sun_ELs):
#         image_path = f"{data_path}/elev_{elev}_azim_{AZ}/images/PerspCam0_000{sample_id}.png"
#         # temp_image_path = f"{tmp_dir_pdf}/temp_image_{i}.png"
#         if os.path.exists(image_path):
#             # convert_to_8bit(image_path, temp_image_path)
#             pdf.add_page()
#             pdf.set_font("Arial", size=12)
#             pdf.cell(200, 10, txt=f"Sun EL = {elev} deg, AZ = {AZ} deg", ln=True, align='C')
#             pdf.image(image_path, x=10, y=20, w=180)  # Adjust the position and size as needed

#     pdf.output(f"{pdf_dest_dir}/id{sample_id}_azim_{AZ}_elevVar.pdf")
#     print(f"'id{sample_id}_azim_{AZ}_elevVar.pdf' PDF created successfully.")

# Function to convert images to 8-bit depth and save them temporarily
def convert_to_8bit(image_path, temp_image_path):
    img = Image.open(image_path)
    img = img.convert('RGB')  # Convert to 8-bit RGB
    img.save(temp_image_path)


# Data paths
map_data_path = "/home/dpisanti/LORNA/mars-sim/dataset/local_test/maps_roughness_1_no-glossy"
query_data_path = "/home/dpisanti/LORNA/mars-sim/dataset/local_test/query_samples_10_400m_roughness_0.5_glossy"

# Define the parameters
samples = 10

sun_AZ_range = [0, 315]  # Define your actual azimuth range (deg)
sun_AZ_step = 45         # Define your actual step (deg)

sun_EL_range = [10, 90]  # Define your actual elevation range (deg)
sun_EL_step = 10         # Define your actual step (deg)

# Generate the list of azimuth angles
sun_AZs = list(range(sun_AZ_range[0], sun_AZ_range[1] + sun_AZ_step, sun_AZ_step))
sun_ELs = list(range(sun_EL_range[0], sun_EL_range[1] + sun_EL_step, sun_EL_step))

# Set output directories
#   for gif
query_gif_out_dir = "./gifs/light_test/queries_rough1_glossy"
map_gif_out_dir = "./gifs/light_test/map_rough0.5_no-glossy"
if not os.path.isdir(query_gif_out_dir):
    os.makedirs(query_gif_out_dir)
if not os.path.isdir(map_gif_out_dir):
    os.makedirs(map_gif_out_dir)
# #   for pdf
# pdf_out_dir = "./pdfs/light_test/queries"
# if not os.path.isdir(pdf_out_dir):
#     os.makedirs(pdf_out_dir)



# GENERATE VARYING ELEVATION
for AZ in sun_AZs:
    create_map_gif_with_varying_elevation(AZ, map_data_path, map_gif_out_dir)
#     for id in range(samples):
#         create_query_gif_with_varying_elevation(AZ, query_data_path, id, query_gif_out_dir)
        
#         # create_pdf_with_varying_elevation(AZ, id, pdf_out_dir)

# GENERATE VARYING AZIMUTH
for EL in sun_ELs:
    create_map_gif_with_varying_azimuth(EL, map_data_path, map_gif_out_dir)
    # for id in range(samples):
    #     create_query_gif_with_varying_azimuth(EL, query_data_path, id, query_gif_out_dir)      
    #     # create_pdf_with_varying_azimuth(EL, id, pdf_out_dir)
        