import os
import pandas as pd
import numpy as np

# Settings
base_dir = "data/keypoints"  # Directory containing Fall and No Fall subdirectories
output_dir = "data/cleaned_keypoints"  # Directory for cleaned CSV files
score_threshold = 0.3  # Confidence score threshold

# Create output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Subdirectories to process
folders = ["fall_keypoints", "no_fall_keypoints"]

for folder in folders:
    input_folder = os.path.join(base_dir, folder)
    output_folder = os.path.join(output_dir, f"cleaned_{folder}")  
    
    # Create subdirectory for cleaned files if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # Retrieve all CSV files in the folder
    csv_files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]

    for csv_file in csv_files:
        input_path = os.path.join(input_folder, csv_file)
        output_path = os.path.join(output_folder, csv_file)

        # Load the CSV file into a DataFrame
        data = pd.read_csv(input_path)

        print(f"Processing file: {input_path}")

        # Step 1: Filter unreliable keypoints
        for col in data.columns:
            if "score" in col:  # Identify confidence score columns
                joint_name = col.replace("score", "").strip()  # Extract joint identifier (e.g., 0, 1, 2)
                x_col = f"x{joint_name}"
                y_col = f"y{joint_name}"

                if x_col in data.columns and y_col in data.columns:
                    # Identify rows where the confidence score is below the threshold
                    mask_below_threshold = data[col] < score_threshold
                    # Set unreliable keypoints (x, y, score) to NaN
                    data.loc[mask_below_threshold, [x_col, y_col, col]] = np.nan  

        # Step 2: Interpolate missing values for smoother data
        # Ensures continuity by estimating missing values based on surrounding points
        data = data.interpolate(method="linear", limit_direction="both")  # Linear interpolation

        # Fill remaining missing values using backward and forward fill
        data = data.bfill().ffill()

        # Step 3: Final cleanup for consistency
        # Replace NaN values in confidence scores with 0 (indicating unreliable keypoints)
        for col in data.columns:
            if "score" in col:
                data[col] = data[col].fillna(0)  

        # Save the cleaned CSV file to the output directory
        data.to_csv(output_path, index=False)
        print(f"Saved cleaned file: {output_path}\n")

print("All files processed and cleaned.")
