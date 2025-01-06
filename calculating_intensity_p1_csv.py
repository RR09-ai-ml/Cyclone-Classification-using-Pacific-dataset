import pandas as pd
"""
    Calculate cyclone intensity based on central pressure and maximum wind speed.

    Parameters:
    - central_pressure: Central pressure in hPa (hectopascals)
    - max_wind_speed: Maximum sustained wind speed in knots

    Returns:
    - A string indicating the cyclone intensity category.
"""
def cyclone_intensity(central_pressure, max_wind_speed):

    # Determine the intensity category based on wind speed
    if max_wind_speed < 34:
        category = "Tropical Depression"
    elif 34 <= max_wind_speed < 64:
        category = "Tropical Storm"
    elif 64 <= max_wind_speed < 83:
        category = "Category 1 Hurricane"
    elif 83 <= max_wind_speed < 96:
        category = "Category 2 Hurricane"
    elif 96 <= max_wind_speed < 113:
        category = "Category 3 Hurricane"
    elif 113 <= max_wind_speed < 137:
        category = "Category 4 Hurricane"
    else:
        category = "Category 5 Hurricane"

    return category


def process_cyclone_data(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Check if the necessary columns exist
    if 'Central Pressure' not in df.columns or 'Max Wind Speed' not in df.columns:
        raise ValueError("CSV must contain 'Central Pressure' and 'Max Wind Speed' columns.")

    # Calculate intensity for each row
    df['Cyclone Intensity'] = df.apply(
        lambda row: cyclone_intensity(row['Central Pressure'], row['Max Wind Speed']),
        axis=1
    )

    # Save the updated DataFrame to a new CSV file
    output_file_path = file_path
    df.to_csv(output_file_path, index=False)
    print(f"Intensity calculated and saved to {output_file_path}")

process_cyclone_data('pacific_data/p1.csv')
