from osgeo import gdal

def tile_dtm(input_file, output_folder, tile_size_x, tile_size_y):
    # Open the original .IMG file with GDAL
    dataset = gdal.Open(input_file)
    
    # Check for PDS driver
    if dataset.GetDriver().ShortName != 'PDS':
        raise ValueError("The file is not recognized as a PDS .IMG file by GDAL.")

    for i in range(0, dataset.RasterXSize, tile_size_x):
        for j in range(0, dataset.RasterYSize, tile_size_y):
            w = min(i + tile_size_x, dataset.RasterXSize) - i
            h = min(j + tile_size_y, dataset.RasterYSize) - j
            tile_filename = f"{output_folder}/tile_{i}_{j}.img"
            # Use GDAL Translate to ensure metadata is preserved
            gdal.Translate(
                tile_filename,
                dataset,
                srcWin=[i, j, w, h],
                format='PDS',  # Specify PDS format
                creationOptions=["PDS_LABEL_TYPE=ODL"]  # Ensure the PVL (ODL) headers are preserved
            )

if __name__ == '__main__':
    input_file = './hirise_assets/jezero_crater/DEM/DTEEC_045994_1985_046060_1985_U01.IMG'
    output_folder = './hirise_assets/jezero_crater/DEM/tiled'
    tile_size_x = 4000  # width of each tile
    tile_size_y = 4000  # height of each tile

    tile_dtm(input_file, output_folder, tile_size_x, tile_size_y)
