# Settings
base_dir = "data/cleaned_keypoints"  # Folder containing cleaned keypoints
output_file = "data/split/training_dataset.csv"  # Output file for the final dataset

# List of folders with labels (1 for fall, 0 for no fall)
folders = [
    ("data/cleaned_keypoints/cleaned_fall_keypoints", 1),  # Fall
    ("data/cleaned_keypoints/cleaned_no_fall_keypoints", 0),  # No fall
]

def create_training_dataset(folders):
    # Initialize a list to store data
    all_data = []
    video_labels = {}  # Dictionary to track labels per video
    expected_columns = None  # Variable to ensure column consistency
    
    # Process each folder (fall and no-fall data)
    for folder, label in folders:
        # Process each file in the folder
        for file in os.listdir(folder):
            if file.endswith(".csv"):  # Process only CSV files
                file_path = os.path.join(folder, file)
                
                # Skip empty files
                if os.stat(file_path).st_size == 0:
                    print(f"Skipping empty file: {file_path}")
                    continue
                
                # Load the CSV file into a DataFrame
                data = pd.read_csv(file_path)
                
                # Initialize expected columns if this is the first file
                if expected_columns is None:
                    expected_columns = set(data.columns)
                
                # Check for column consistency
                if set(data.columns) != expected_columns:
                    print(f"Skipping file due to column mismatch: {file_path}")
                    continue
                
                # Add 'label' column to indicate fall (1) or no fall (0)
                data['label'] = label
                data['video_file'] = os.path.join(folder, file)  # Add full file path
                # Append data to the list
                all_data.append(data)
                # Track the label for the video (full path)
                video_labels[file_path] = label

    # Merge all data into a single DataFrame
    training_data = pd.concat(all_data, ignore_index=True)
    return training_data, video_labels

# Create the dataset
training_data, video_labels = create_training_dataset(folders)

# Save the final dataset
training_data.to_csv(output_file, index=False)
print(f"Dataset saved to {output_file}")

# Extract unique video files (full paths)
unique_videos = [os.path.join(folder, file) for folder, _ in folders for file in os.listdir(folder) if file.endswith(".csv")]
video_labels_list = [video_labels[v] for v in unique_videos if v in video_labels]

# Split into Train and Test sets (stratified based on labels)
train_videos, test_videos = train_test_split(
    unique_videos, test_size=0.2, random_state=42, stratify=video_labels_list
)

# Ensure no overlap between Train and Test sets
assert set(train_videos).isdisjoint(test_videos), "Train and Test sets contain overlapping videos!"

print("No overlapping videos found.")

# Filter data based on video file names (ensuring no overlap)
train_data = training_data[training_data['video_file'].isin(train_videos)]
test_data = training_data[training_data['video_file'].isin(test_videos)]

# Save the Train and Test data
train_data.to_csv("train_data.csv", index=False)
test_data.to_csv("test_data.csv", index=False)

# Prepare features (X) and labels (y)
X_train = train_data.drop(columns=["label", "video_file"])  # Features for training
y_train = train_data["label"]  # Labels for training
X_test = test_data.drop(columns=["label", "video_file"])  # Features for testing
y_test = test_data["label"]  # Labels for testing

# Save Train and Test datasets
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

# Save lists of Train/Test videos
pd.DataFrame(train_videos, columns=["video_file"]).to_csv("train_videos.csv", index=False)
pd.DataFrame(test_videos, columns=["video_file"]).to_csv("test_videos.csv", index=False)

print("Training and test sets saved.")
