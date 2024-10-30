import geopandas as gpd
import os

# Path to the shapefile
shapefile_path = r"C:\GIS\Elevation_plots.zip"

# Check if the shapefile exists
if not os.path.exists(shapefile_path):
    print(f"Shapefile does not exist at: {shapefile_path}")
else:
    try:
        # Attempt to read the shapefile
        gdf = gpd.read_file(shapefile_path)
        print("Shapefile read successfully!")
        print(gdf.head())  # Display the first few rows
    except Exception as e:
        print(f"An error occurred while reading the shapefile: {e}")
