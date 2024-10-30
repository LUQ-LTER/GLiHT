import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ---------------------------- Configuration ---------------------------- #

# Path to the output CSV file from the analysis
OUTPUT_CSV = r"C:\Users\ml1451\OneDrive - USNH\GIS\GLiHT\vegetation_structure_extraction_pivoted_v1.csv"

# Output directory for saving plots
PLOT_OUTPUT_DIR = r"C:\Users\ml1451\OneDrive - USNH\GIS\GLiHT\Plots"

# Plots to exclude
EXCLUDE_PLOTS = ['Elev950', 'Elev1000']

# Define class breaks as per custom_class_breaks()
class_breaks = [0, 2.5, 5, 10, 15, 20, 30, 50]  # Ensure this matches your custom class breaks

# ---------------------------- Ensure Output Directory Exists ---------------------------- #

if not os.path.exists(PLOT_OUTPUT_DIR):
    os.makedirs(PLOT_OUTPUT_DIR)
    print(f"Created directory for plots at: {PLOT_OUTPUT_DIR}")

# ---------------------------- Read and Prepare Data ---------------------------- #

# Read the CSV file
try:
    df = pd.read_csv(OUTPUT_CSV)
    print(f"Successfully read the CSV file from {OUTPUT_CSV}")
except Exception as e:
    print(f"Error reading the CSV file: {e}")
    exit(1)

# Exclude specified plots
df_filtered = df[~df['plot'].isin(EXCLUDE_PLOTS)]
print(f"Excluded plots: {EXCLUDE_PLOTS}")
print(f"Number of plots after exclusion: {df_filtered['plot'].nunique()}")

# Identify class percentage columns dynamically (assuming classes 1 to 5)
class_percentage_cols = [col for col in df_filtered.columns if 'class_' in col and 'percentage' in col]

# Exclude classes beyond class 5 (if they exist)
class_percentage_cols = [col for col in class_percentage_cols if any(f'class_{i}_percentage' == col for i in range(1,8))]

print(f"Class percentage columns to be plotted: {class_percentage_cols}")

# Melt the DataFrame to long format for easier plotting
df_melted = df_filtered.melt(
    id_vars=['plot', 'year'],  # Ensure 'year' column exists
    value_vars=class_percentage_cols,
    var_name='Class_Percentage',
    value_name='Percentage'
)

# Extract class number from 'Class_Percentage'
df_melted['Class'] = df_melted['Class_Percentage'].str.extract(r'class_(\d+)_percentage').astype(int)

# Map Class numbers to labels with height ranges
class_labels = []
for i in range(1, 8):
    lower = class_breaks[i - 1]
    upper = class_breaks[i]
    class_labels.append(f"Class {i}\n({lower}-{upper} m)")

class_label_dict = {i: label for i, label in zip(range(1,8), class_labels)}
df_melted['Class_Label'] = df_melted['Class'].map(class_label_dict)

# Clean up 'Class_Percentage' column
df_melted.drop(columns=['Class_Percentage'], inplace=True)

# ---------------------------- Plotting ---------------------------- #

# Set the visual style
sns.set(style="whitegrid")

# Loop over each plot and create a separate plot
plots = df_melted['plot'].unique()

for plot_name in plots:
    df_plot = df_melted[df_melted['plot'] == plot_name]

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df_plot,
        x='Class_Label',
        y='Percentage',
        hue='year',
        palette='viridis'
    )

    plt.title(f"Canopy Height Class Percentages for Plot {plot_name}")
    plt.xlabel("Canopy Height Classes")
    plt.ylabel("Percentage")
    plt.legend(title='Year')
    plt.xticks(rotation=45)

    # Save the plot
    plot_file_path = os.path.join(PLOT_OUTPUT_DIR, f"{plot_name}_canopy_height_classification.png")
    plt.tight_layout()
    plt.savefig(plot_file_path)
    plt.close()
    print(f"Plot saved for plot {plot_name} at {plot_file_path}")

print("All plots have been generated and saved.")
