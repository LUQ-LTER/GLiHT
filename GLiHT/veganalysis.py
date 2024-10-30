import geopandas as gpd
import rasterio
from rasterstats import zonal_stats
import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm
import mapclassify
from rasterio.mask import mask as rio_mask
from shapely.geometry import box
import logging
import re
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------- Configuration ---------------------------- #
sdf
# Path to the Elevation_plots.shp shapefile
SHAPEFILE_PATH = r"C:\Users\ml1451\OneDrive - USNH\GIS\GLiHT\Elevation_plots.shp"

# Base directory containing GLiHT derived data for all years
BASE_RASTER_DIR = r"C:\Users\ml1451\OneDrive - USNH\GIS\GLiHT\GLiHT derived"

# Years to process
YEARS = [2017, 2018, 2020]

# Mapping of vegetation structure factors to their respective file name patterns
VARIABLES = {
    'CHM': '*_CHM.tif',
    'CHM_rugosity': '*_chm_rugosity.tif',
    'slope': '*_slope.tif',
    'aspect': '*_aspect.tif',
    'DSM': '*_DSM.tif',
    'DSM_rugosity': '*_dsm_rugosity.tif'
}

# Output CSV file path
OUTPUT_CSV = r"C:\Users\ml1451\OneDrive - USNH\GIS\GLiHT\vegetation_structure_extraction.csv"
OUTPUT_PIVOTED_CSV = os.path.splitext(OUTPUT_CSV)[0] + '_pivoted.csv'

# Height threshold for canopy density (in meters)
CANOPY_DENSITY_THRESHOLD = 3  # Adjust as needed

# Maximum number of classes for canopy height classification
MAX_CLASSES = 5

# ---------------------------- Logging Configuration ---------------------------- #

# Configure logging
logging.basicConfig(
    filename='vegetation_analysis.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)


# ---------------------------- Helper Functions ---------------------------- #

def find_raster_files(raster_dir, pattern):
    """
    Finds all raster files in the specified directory matching the given pattern.

    :param raster_dir: Directory to search for raster files.
    :param pattern: Glob pattern to match raster files.
    :return: List of file paths matching the pattern.
    """
    search_pattern = os.path.join(raster_dir, pattern)
    return glob.glob(search_pattern, recursive=True)

def collect_chm_values(raster_files, bad_value=None, max_threshold=None):
    """
    Collects CHM values from multiple raster files, excluding bad values and applying a maximum threshold.

    :param raster_files: List of CHM raster file paths.
    :param bad_value: Specific CHM value to exclude (e.g., 52.96).
    :param max_threshold: Maximum allowable CHM value to include.
    :return: Numpy array of collected CHM values.
    """
    chm_values = []
    for raster_path in raster_files:
        try:
            with rasterio.open(raster_path) as src:
                chm_data = src.read(1)
                # Replace nodata with NaN
                if src.nodata is not None:
                    chm_data = np.where(chm_data == src.nodata, np.nan, chm_data)
                # Flatten the array
                chm_flat = chm_data.flatten()
                # Remove NaNs
                chm_flat = chm_flat[~np.isnan(chm_flat)]
                # Exclude bad values
                if bad_value is not None:
                    chm_flat = chm_flat[chm_flat != bad_value]
                # Apply maximum threshold
                if max_threshold is not None:
                    chm_flat = chm_flat[chm_flat <= max_threshold]
                # Append to the list
                chm_values.append(chm_flat)
        except Exception as e:
            print(f"Error reading CHM raster {raster_path}: {e}")
            logging.error(f"Error reading CHM raster {raster_path}: {e}")
            continue
    # Concatenate all values into a single array
    if chm_values:
        return np.concatenate(chm_values)
    else:
        return np.array([])


def determine_natural_breaks(chm_values, max_classes=6):
    """
    Determines natural breaks (Jenks) for CHM values using the specified maximum number of classes.

    :param chm_values: Numpy array of CHM pixel values.
    :param max_classes: Maximum number of classes to determine.
    :return: List of class boundaries.
    """
    # Ensure there are enough unique values to determine breaks
    unique_values = np.unique(chm_values)
    if len(unique_values) < max_classes:
        print("Warning: Not enough unique CHM values to determine the desired number of classes.")
        logging.warning("Not enough unique CHM values to determine the desired number of classes.")
        max_classes = len(unique_values)

    if max_classes < 2:
        print("Error: At least two classes are required for classification.")
        logging.error("At least two classes are required for classification.")
        return np.array([])

    classifier = mapclassify.NaturalBreaks(y=chm_values, k=max_classes)
    breaks = classifier.bins
    print(f"Determined natural breaks: {breaks}")
    logging.info(f"Determined natural breaks: {breaks}")
    return breaks


def determine_quantile_breaks(chm_values, num_classes=5):
    """
    Determines class breaks based on quantiles.

    :param chm_values: Numpy array of CHM pixel values.
    :param num_classes: Number of classes.
    :return: List of class boundaries.
    """
    quantiles = np.linspace(0, 1, num_classes + 1)
    breaks = np.quantile(chm_values, quantiles)
    print(f"Determined quantile-based breaks: {breaks}")
    logging.info(f"Determined quantile-based breaks: {breaks}")
    return breaks.tolist()


def get_custom_class_breaks():
    """
    Defines custom class breaks based on data analysis.

    :return: List of class boundary values.
    """
    # Example custom breaks; adjust based on your data analysis
    return [0, 2.5, 5, 10, 15, 20, 30, 50]  # Ensure the last value covers all CHM values


def classify_canopy_density(chm_data, class_breaks):
    """
    Classifies CHM data into defined classes and calculates the percentage of canopy in each class.

    :param chm_data: Numpy array of CHM values.
    :param class_breaks: List of class boundary values.
    :return: Dictionary with class labels as keys and percentage of canopy as values.
    """
    if chm_data is None or len(chm_data) == 0:
        return {f'class_{i}_percentage': np.nan for i in range(1, len(class_breaks))}

    # Define class labels based on number of breaks
    num_classes = len(class_breaks) - 1  # Since breaks define intervals between classes

    # Digitize assigns integers to bins; classes start at 1
    class_indices = np.digitize(chm_data, class_breaks, right=False)

    # Calculate percentages
    total_pixels = len(chm_data)
    percentages = {}
    for i in range(1, num_classes + 1):
        count = np.sum(class_indices == i)
        percentage = (count / total_pixels) * 100 if total_pixels > 0 else np.nan
        percentages[f'class_{i}_percentage'] = percentage

    return percentages


def extract_chm_class_percentages(gdf, raster_path, class_breaks, year, bad_value=None, max_threshold=None):
    """
    Extracts canopy class percentages for each plot from a CHM raster,
    excluding specified bad values or values exceeding a maximum threshold.

    :param gdf: GeoDataFrame containing plot geometries.
    :param raster_path: Path to the CHM raster file.
    :param class_breaks: List of class boundary values.
    :param year: The year being processed (e.g., 2017).
    :param bad_value: Specific CHM value to exclude (e.g., 52.96).
    :param max_threshold: Maximum allowable CHM value to include.
    :return: Dictionary with plot indices as keys and class percentage dictionaries as values.
    """
    class_percentages = {}
    try:
        with rasterio.open(raster_path) as src:
            for idx, row in gdf.iterrows():
                geometry = row['geometry']
                plot_name = row['plot']
                try:
                    clipped_data, clipped_transform = rio_mask(src, [geometry], crop=True, nodata=src.nodata,
                                                               filled=False)
                    chm_values = clipped_data[0]

                    # Check if clipped_data is a masked array
                    if isinstance(chm_values, np.ma.MaskedArray):
                        chm_flat = chm_values.compressed()
                    else:
                        # Replace nodata with NaN if nodata is defined
                        if src.nodata is not None:
                            chm_flat = chm_values[chm_values != src.nodata].flatten()
                        else:
                            chm_flat = chm_values.flatten()

                    # Remove any remaining NaNs
                    chm_flat = chm_flat[~np.isnan(chm_flat)]

                    # Exclude bad values
                    if bad_value is not None:
                        chm_flat = chm_flat[chm_flat != bad_value]

                    # Exclude values above threshold
                    if max_threshold is not None:
                        chm_flat = chm_flat[chm_flat <= max_threshold]

                    if chm_flat.size == 0:
                        percentages = {f'class_{i}_percentage': np.nan for i in range(1, len(class_breaks) + 2)}
                    else:
                        # Classify canopy density
                        canopy_percentages = classify_canopy_density(chm_flat, class_breaks)
                        percentages = canopy_percentages

                except Exception as e:
                    print(f"Error processing plot '{plot_name}': {e}")
                    logging.error(f"Error processing plot '{plot_name}': {e}")
                    percentages = {f'class_{i}_percentage': np.nan for i in range(1, len(class_breaks) + 2)}

                class_percentages[idx] = percentages
    except Exception as e:
        print(f"Error opening raster {raster_path}: {e}")
        logging.error(f"Error opening raster {raster_path}: {e}")
        for idx in gdf.index:
            class_percentages[idx] = {f'class_{i}_percentage': np.nan for i in range(1, len(class_breaks) + 2)}

    return class_percentages


def check_overlap(gdf, raster_path):
    """
    Checks if any geometries in the GeoDataFrame overlap with the raster extent.

    :param gdf: GeoDataFrame containing plot geometries.
    :param raster_path: Path to the raster file.
    :return: Boolean indicating if any overlaps exist.
    """
    with rasterio.open(raster_path) as src:
        raster_bounds = src.bounds
        raster_crs = src.crs
    # Reproject shapefile to raster CRS if not already
    if gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)
    # Create a shapely box from raster bounds
    raster_bbox = box(raster_bounds.left, raster_bounds.bottom, raster_bounds.right, raster_bounds.top)
    # Check for overlaps
    overlaps = gdf.intersects(raster_bbox)
    return overlaps.any()


def plot_chm_distribution(chm_values, year):
    plt.figure(figsize=(10,6))
    plt.hist(chm_values, bins=50, color='skyblue', edgecolor='black')
    plt.title(f'CHM Value Distribution for {year}')
    plt.xlabel('Canopy Height (m)')
    plt.ylabel('Frequency')
    plt.show()


def print_summary_statistics(chm_values, year):
    print(f"\nSummary Statistics for CHM in {year}:")
    print(f"Mean: {np.mean(chm_values):.2f} m")
    print(f"Median: {np.median(chm_values):.2f} m")
    print(f"Standard Deviation: {np.std(chm_values):.2f} m")
    print(f"Min: {np.min(chm_values):.2f} m")
    print(f"Max: {np.max(chm_values):.2f} m")

def pivot_csv(input_csv, output_pivoted_csv):
    """
    Pivots the CSV from wide to long format by year.

    :param input_csv: Path to the input wide-format CSV.
    :param output_pivoted_csv: Path to save the pivoted long-format CSV.
    """
    # Read the wide-format CSV
    df_wide = pd.read_csv(input_csv)

    # Optional: Standardize column names (strip spaces)
    df_wide.columns = df_wide.columns.str.strip()

    # Define patterns to identify metric and class percentage columns
    metric_pattern = re.compile(r'^(CHM|CHM_rugosity|slope|aspect|DSM|DSM_rugosity)_(mean|std)_(2017|2018|2020)$')
    class_pattern = re.compile(r'^class_(\d+)_percentage_(2017|2018|2020)$')

    # Identify metric and class percentage columns
    metric_cols = [col for col in df_wide.columns if metric_pattern.match(col)]
    class_cols = [col for col in df_wide.columns if class_pattern.match(col)]

    # Melt metrics
    df_metrics = df_wide.melt(
        id_vars=['Id', 'plot', 'lat', 'lon', 'geometry'],
        value_vars=metric_cols,
        var_name='metric_year',
        value_name='metric_value'
    )

    # Extract metric name and year
    df_metrics[['metric', 'stat', 'year']] = df_metrics['metric_year'].str.extract(r'^(CHM|CHM_rugosity|slope|aspect|DSM|DSM_rugosity)_(mean|std)_(\d{4})$')
    df_metrics.drop(columns=['metric_year'], inplace=True)

    # Pivot metrics to wide again with 'metric' as separate columns
    df_metrics_pivot = df_metrics.pivot_table(
        index=['Id', 'plot', 'lat', 'lon', 'geometry', 'year'],
        columns='metric',
        values='metric_value'
    ).reset_index()

    # Melt class percentages
    df_classes = df_wide.melt(
        id_vars=['Id', 'plot', 'lat', 'lon', 'geometry'],
        value_vars=class_cols,
        var_name='class_year',
        value_name='class_percentage'
    )

    # Extract class number and year
    df_classes[['class', 'year']] = df_classes['class_year'].str.extract(r'^class_(\d+)_percentage_(\d{4})$')
    df_classes.drop(columns=['class_year'], inplace=True)

    # Pivot class percentages to wide again with 'class' as prefix
    df_classes_pivot = df_classes.pivot_table(
        index=['Id', 'plot', 'lat', 'lon', 'geometry', 'year'],
        columns='class',
        values='class_percentage'
    ).reset_index()

    # Get list of class numbers as strings
    class_numbers = df_classes['class'].unique().tolist()

    # Rename only the class number columns
    df_classes_pivot.rename(columns=lambda x: f'class_{x}_percentage' if x in class_numbers else x, inplace=True)

    # Merge metrics and class percentages on common columns including 'Id'
    df_final = pd.merge(df_metrics_pivot, df_classes_pivot, on=['Id', 'plot', 'lat', 'lon', 'geometry', 'year'], how='outer')

    # Optional: Reorder columns for better readability
    cols_order = ['Id', 'plot', 'lat', 'lon', 'geometry', 'year'] + \
                 [col for col in df_final.columns if col in ['CHM', 'CHM_rugosity', 'slope', 'aspect', 'DSM', 'DSM_rugosity']] + \
                 [col for col in df_final.columns if 'class_' in col and 'percentage' in col]
    df_final = df_final[cols_order]

    # Save the pivoted DataFrame to CSV
    df_final.to_csv(output_pivoted_csv, index=False)
    print(f"Pivoted data saved to {output_pivoted_csv}")
    logging.info(f"Pivoted data saved to {output_pivoted_csv}")

def validate_percentages(df_pivoted):
    """
    Validates that class percentages sum to ~100% for each plot and year.
    """
    class_cols = [col for col in df_pivoted.columns if 'class_' in col and 'percentage' in col]
    if not class_cols:
        class_cols = [col for col in df_pivoted.columns if 'class_' in col and 'percentage' in col]
    df_pivoted['total_percentage'] = df_pivoted[class_cols].sum(axis=1)
    discrepancies = df_pivoted[~df_pivoted['total_percentage'].between(99, 101)]
    if not discrepancies.empty:
        print("Warning: Some plots have total class percentages not summing to ~100%.")
        print(discrepancies[['Id', 'plot', 'year', 'total_percentage']])
        logging.warning("Some plots have total class percentages not summing to ~100%.")
    else:
        print("All plots have class percentages summing to approximately 100%.")


# ---------------------------- Main Processing ---------------------------- #

def main():
    print("Starting vegetation structure extraction and analysis process...")
    logging.info("Starting vegetation structure extraction and analysis process.")

    # Check if shapefile exists
    if not os.path.exists(SHAPEFILE_PATH):
        print(f"Shapefile does not exist at: {SHAPEFILE_PATH}")
        logging.error(f"Shapefile does not exist at: {SHAPEFILE_PATH}")
        return

    # Read the shapefile using GeoPandas
    print(f"Reading shapefile from {SHAPEFILE_PATH}...")
    logging.info(f"Reading shapefile from {SHAPEFILE_PATH}...")
    try:
        plots_gdf = gpd.read_file(SHAPEFILE_PATH)
    except Exception as e:
        print(f"Error reading shapefile: {e}")
        logging.error(f"Error reading shapefile: {e}")
        return

    # Check if 'plot' field exists
    if 'plot' not in plots_gdf.columns:
        print("Error: The shapefile does not contain a 'plot' field.")
        logging.error("The shapefile does not contain a 'plot' field.")
        print("Please ensure that the shapefile has a 'plot' field representing the plot names.")
        return
    else:
        print("'plot' column found in shapefile.")
        logging.info("'plot' column found in shapefile.")

    # Ensure that the 'plot' field is unique or handle duplicates as needed
    if plots_gdf['plot'].duplicated().any():
        print("Warning: Duplicate plot names found in the 'plot' field. Consider ensuring unique plot names.")
        logging.warning("Duplicate plot names found in the 'plot' field.")
    else:
        print("All plot names are unique.")
        logging.info("All plot names are unique.")

    # Verify Geometry Types
    print("Geometry Types in GeoDataFrame:")
    print(plots_gdf.geometry.type.unique())
    logging.info(f"Geometry Types: {plots_gdf.geometry.type.unique()}")

    # If geometries are Points, buffer them to create polygons
    if plots_gdf.geometry.type.unique()[0] == 'Point':
        print("Shapefile contains Point geometries. Buffering points to create Polygonal plots...")
        logging.info("Shapefile contains Point geometries. Buffering points to create Polygonal plots...")
        # Define a buffer distance (in meters). Adjust based on actual plot sizes.
        buffer_distance = 10  # Example: 10 meters
        plots_gdf['geometry'] = plots_gdf.geometry.buffer(buffer_distance)
        print("Buffering completed.")
        logging.info("Buffering completed.")
        print("New Geometry Types after Buffering:")
        print(plots_gdf.geometry.type.unique())
        logging.info(f"New Geometry Types after Buffering: {plots_gdf.geometry.type.unique()}")

    # ---------------------------- Step 1: Determine Class Breaks ---------------------------- #

    print("\nDetermining class breaks using 2017 CHM data...")
    logging.info("Determining class breaks using 2017 CHM data...")

    # Find all 2017 CHM raster files
    raster_dir_2017 = os.path.join(BASE_RASTER_DIR, "Derived 2017")
    chm_pattern = VARIABLES['CHM']
    chm_raster_files_2017 = find_raster_files(raster_dir_2017, chm_pattern)

    if not chm_raster_files_2017:
        print("Error: No 2017 CHM raster files found.")
        logging.error("No 2017 CHM raster files found.")
        return

    # Collect all CHM values from 2017
    # You need to define the 'collect_chm_values' function
    chm_values_2017 = collect_chm_values(chm_raster_files_2017, bad_value=52.96, max_threshold=50.0)

    if chm_values_2017.size == 0:
        print("Error: No valid CHM data found in 2017 rasters.")
        logging.error("No valid CHM data found in 2017 rasters.")
        return

    # Determine class breaks using custom breaks
    class_breaks = get_custom_class_breaks()
    print(f"Using custom class breaks: {class_breaks}")
    logging.info(f"Using custom class breaks: {class_breaks}")

    # Save class breaks for reference
    breaks_df = pd.DataFrame({'Class_Breaks': class_breaks})
    breaks_csv = os.path.splitext(OUTPUT_CSV)[0] + '_class_breaks.csv'
    breaks_df.to_csv(breaks_csv, index=False)
    print(f"Class breaks saved to {breaks_csv}")
    logging.info(f"Class breaks saved to {breaks_csv}")

    # ---------------------------- Step 2: Extract Metrics and Class Percentages ---------------------------- #

    print("\nExtracting metrics and canopy class percentages for each plot and year...")
    logging.info("Extracting metrics and canopy class percentages for each plot and year...")

    # Initialize a nested dictionary to store class percentages
    # Structure: {year: {plot_idx: {class_1_percentage: [values], ...}}, ...}
    class_percentages_dict = {
        year: {
            plot_idx: {f'class_{i}_percentage': [] for i in range(1, len(class_breaks))}
            for plot_idx in plots_gdf.index
        }
        for year in YEARS
    }

    for year in YEARS:
        print(f"\nProcessing year: {year}")
        logging.info(f"Processing year: {year}")

        # Define the raster directory for the current year
        raster_dir = os.path.join(BASE_RASTER_DIR, f"Derived {year}")

        if not os.path.exists(raster_dir):
            print(f"Raster directory for year {year} does not exist: {raster_dir}")
            logging.warning(f"Raster directory for year {year} does not exist: {raster_dir}")
            continue

        # Iterate through each vegetation structure factor
        for var_name, pattern in VARIABLES.items():
            print(f"  Extracting {var_name}...")
            logging.info(f"  Extracting {var_name}...")

            # Find all raster files matching the pattern
            raster_files = find_raster_files(raster_dir, pattern)

            if not raster_files:
                print(f"    No raster files found for {var_name} in year {year}.")
                logging.warning(f"No raster files found for {var_name} in year {year}.")
                # Assign NaNs for this variable
                plots_gdf[f"{var_name}_mean_{year}"] = np.nan
                plots_gdf[f"{var_name}_std_{year}"] = np.nan
                if var_name == 'CHM':
                    # For CHM, also assign NaNs for canopy class percentages
                    num_classes = len(class_breaks) - 1
                    for i in range(1, num_classes + 1):
                        plots_gdf[f'class_{i}_percentage_{year}'] = np.nan
                continue

            # Check if any raster overlaps with shapefile
            overlaps = any([check_overlap(plots_gdf, raster) for raster in raster_files])
            if not overlaps:
                print(f"    No overlapping rasters found for {var_name} in year {year}.")
                logging.warning(f"No overlapping rasters found for {var_name} in year {year}.")
                # Assign NaNs as before
                plots_gdf[f"{var_name}_mean_{year}"] = np.nan
                plots_gdf[f"{var_name}_std_{year}"] = np.nan
                if var_name == 'CHM':
                    num_classes = len(class_breaks) - 1
                    for i in range(1, num_classes + 1):
                        plots_gdf[f'class_{i}_percentage_{year}'] = np.nan
                continue

            # Initialize lists to collect metrics
            means = []
            stds = []

            # Process each raster file
            for raster_path in raster_files:
                print(f"    Processing raster: {os.path.basename(raster_path)}")
                logging.info(f"    Processing raster: {os.path.basename(raster_path)}")
                try:
                    with rasterio.open(raster_path) as src:
                        raster_crs = src.crs
                        # Reproject shapefile to raster CRS if not already
                        if plots_gdf.crs != raster_crs:
                            gdf_reproj = plots_gdf.to_crs(raster_crs)
                            print(f"      Reprojected shapefile to {raster_crs}")
                            logging.info(f"      Reprojected shapefile to {raster_crs}")
                        else:
                            gdf_reproj = plots_gdf

                        # Perform zonal statistics for mean and std
                        stats = zonal_stats(
                            gdf_reproj,
                            raster_path,
                            stats=["mean", "std"],
                            nodata=src.nodata,
                            geojson_out=False
                        )
                        means_list = [stat['mean'] if stat['mean'] is not None else np.nan for stat in stats]
                        stds_list = [stat['std'] if stat['std'] is not None else np.nan for stat in stats]
                        means.append(means_list)
                        stds.append(stds_list)

                        # If CHM, calculate canopy class percentages
                        if var_name == 'CHM':
                            percentages_dict = extract_chm_class_percentages(
                                gdf_reproj, raster_path, class_breaks, year, bad_value=52.96, max_threshold=50.0
                            )
                            for plot_idx, percentages in percentages_dict.items():
                                for key, value in percentages.items():
                                    # Append the percentage to the corresponding class list
                                    class_percentages_dict[year][plot_idx][key].append(value)

                except Exception as e:
                    print(f"    Error processing raster {raster_path}: {e}")
                    logging.error(f"    Error processing raster {raster_path}: {e}")
                    # Assign NaNs for this raster
                    means.append([np.nan] * len(plots_gdf))
                    stds.append([np.nan] * len(plots_gdf))
                    if var_name == 'CHM':
                        for i in range(1, len(class_breaks)):
                            for plot_idx in plots_gdf.index:
                                class_percentages_dict[year][plot_idx][f'class_{i}_percentage'].append(np.nan)
                    continue

            # Aggregate metrics by averaging across multiple rasters
            if means:
                means_array = np.array(means)
                if means_array.size > 0 and not np.all(np.isnan(means_array)):
                    mean_final = np.nanmean(means_array, axis=0)
                else:
                    mean_final = np.full(len(plots_gdf), np.nan)
                plots_gdf[f"{var_name}_mean_{year}"] = mean_final
            else:
                plots_gdf[f"{var_name}_mean_{year}"] = np.nan

            if stds:
                stds_array = np.array(stds)
                if stds_array.size > 0 and not np.all(np.isnan(stds_array)):
                    std_final = np.nanmean(stds_array, axis=0)
                else:
                    std_final = np.full(len(plots_gdf), np.nan)
                plots_gdf[f"{var_name}_std_{year}"] = std_final
            else:
                plots_gdf[f"{var_name}_std_{year}"] = np.nan

            # Assign canopy class percentages by computing the mean across all rasters
            if var_name == 'CHM':
                for plot_idx in plots_gdf.index:
                    for i in range(1, len(class_breaks)):
                        class_key = f'class_{i}_percentage'
                        percentages = class_percentages_dict[year][plot_idx][class_key]
                        if len(percentages) > 0 and not np.all(np.isnan(percentages)):
                            mean_percentage = np.nanmean(percentages)
                        else:
                            mean_percentage = np.nan
                        plots_gdf.at[plot_idx, f'{class_key}_{year}'] = mean_percentage

                print(f"    Extracted canopy class percentages for {var_name} in year {year}.")
                logging.info(f"    Extracted canopy class percentages for {var_name} in year {year}.")

            print(f"    Extracted metrics for {var_name} in year {year}.")
            logging.info(f"    Extracted metrics for {var_name} in year {year}.")

    # ---------------------------- Step 3: Compile and Save Results ---------------------------- #

        print("\nCompiling results...")
        logging.info("Compiling results...")

        # Select relevant columns: 'Id', 'plot', 'lat', 'lon', 'geometry', and all vegetation factor columns
        veg_columns = [col for col in plots_gdf.columns if any(var in col for var in VARIABLES.keys())]
        # Also include canopy class percentages
        class_percentage_columns = [col for col in plots_gdf.columns if 'class_' in col and 'percentage' in col]
        final_columns = ['Id', 'plot', 'lat', 'lon', 'geometry'] + veg_columns + class_percentage_columns
        final_df = plots_gdf[final_columns].copy()

        # Debugging: Check selected columns
        print("Selected columns for final DataFrame:")
        print(final_columns)
        logging.info(f"Selected columns for final DataFrame: {final_columns}")

        # Save the main wide-format CSV
        print("Saving results to CSV in wide format...")
        logging.info("Saving results to CSV in wide format...")
        try:
            final_df.to_csv(OUTPUT_CSV, index=False)
            print("Data extraction and compilation completed successfully.")
            logging.info("Data extraction and compilation completed successfully.")
        except Exception as e:
            print(f"Error saving results to CSV: {e}")
            logging.error(f"Error saving results to CSV: {e}")

        # ---------------------------- Step 4: Pivot the CSV ---------------------------- #

        print("\nPivoting the CSV to longer format by year...")
        logging.info("Pivoting the CSV to longer format by year...")
        pivot_csv(OUTPUT_CSV, OUTPUT_PIVOTED_CSV)

        # ---------------------------- Step 5: Validate the Pivoted Data ---------------------------- #

        print("\nValidating pivoted data...")
        logging.info("Validating pivoted data...")

        # Read the pivoted CSV
        df_pivoted = pd.read_csv(OUTPUT_PIVOTED_CSV)

        # Validate that class percentages sum to ~100%
        class_cols = [col for col in df_pivoted.columns if 'class_' in col and 'percentage' in col]
        df_pivoted['total_percentage'] = df_pivoted[class_cols].sum(axis=1)
        discrepancies = df_pivoted[~df_pivoted['total_percentage'].between(99, 101)]

        if not discrepancies.empty:
            print("Warning: Some plots have total class percentages not summing to ~100%.")
            print(discrepancies[['Id', 'plot', 'year', 'total_percentage']])
            logging.warning("Some plots have total class percentages not summing to ~100%.")

            # Optional: Save discrepancies to a separate CSV for review
            discrepancies_csv = os.path.splitext(OUTPUT_PIVOTED_CSV)[0] + '_discrepancies.csv'
            discrepancies.to_csv(discrepancies_csv, index=False)
            print(f"Discrepancies saved to {discrepancies_csv}")
            logging.info(f"Discrepancies saved to {discrepancies_csv}")
        else:
            print("All plots have class percentages summing to approximately 100%.")
            logging.info("All plots have class percentages summing to approximately 100%.")

        # ---------------------------- Step 6: Additional Analysis ---------------------------- #

        print("\nPerforming additional analyses...")
        logging.info("Performing additional analyses...")

        # Example: Calculate correlation between slope and CHM_mean for each year
        correlation_results = []
        for year in YEARS:
            slope_mean_col = f'slope_mean'
            chm_mean_col = f'CHM_mean'
            # Filter data for the specific year
            df_year = df_pivoted[df_pivoted['year'] == year]
            if slope_mean_col in df_year.columns and chm_mean_col in df_year.columns:
                correlation = df_year[slope_mean_col].corr(df_year[chm_mean_col])
                correlation_results.append({'Year': year, 'Slope_CHM_Correlation': correlation})
            else:
                correlation_results.append({'Year': year, 'Slope_CHM_Correlation': np.nan})

        correlation_df = pd.DataFrame(correlation_results)
        print("Slope and CHM Correlation by Year:")
        print(correlation_df)
        logging.info(f"Slope and CHM Correlation by Year:\n{correlation_df}")

        # Save correlation results
        correlation_csv = os.path.splitext(OUTPUT_CSV)[0] + '_correlations.csv'
        try:
            correlation_df.to_csv(correlation_csv, index=False)
            print(f"Correlation results saved to {correlation_csv}.")
            logging.info(f"Correlation results saved to {correlation_csv}.")
        except Exception as e:
            print(f"Error saving correlation results to CSV: {e}")
            logging.error(f"Error saving correlation results to CSV: {e}")

        print("\nAdditional analyses completed.")
        logging.info("Additional analyses completed.")

if __name__ == "__main__":
    main()

