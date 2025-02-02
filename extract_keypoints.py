import tensorflow as tf
import numpy as np
import cv2
import os
import csv

# Initialize the MoveNet model for keypoint detection
interpreter = tf.lite.Interpreter(model_path=r"C:\Users\d2907\file2\3.tflite")
interpreter.allocate_tensors()

# Function to extract keypoints from a single video frame
def extract_keypoints(frame):
    # Resize the image while maintaining the aspect ratio
    img = tf.image.resize_with_pad(np.expand_dims(frame, axis=0), 256, 256)
    input_img = tf.cast(img, dtype=tf.float32)

    # Retrieve input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set the input tensor and invoke the model
    interpreter.set_tensor(input_details[0]['index'], np.array(input_img))
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

    # Return the keypoints and their confidence scores
    return np.squeeze(keypoints_with_scores)

# Function to process a single video and calculate the percentage of keypoints with high confidence
def process_video(video_path, output_csv_path=None):
    cap = cv2.VideoCapture(video_path)
    keypoints_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Extract keypoints from the frame
        keypoints = extract_keypoints(frame)
        keypoints_list.append(keypoints)

        # If we're writing to CSV, save the keypoints of this frame
        if output_csv_path is not None:
            with open(output_csv_path, mode='a', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                # Write the video name and all 17 keypoints with x, y, and confidence
                row = [video_path]
                for keypoint in keypoints:
                    row.extend([keypoint[0], keypoint[1], keypoint[2]])  # x, y, score for each keypoint
                csv_writer.writerow(row)

    cap.release()

    # Extract confidence scores from the keypoints (third value in each keypoint set)
    confidence_scores = np.array(keypoints_list)[:, :, 2]
    high_confidence = (confidence_scores > 0.4).sum()  # Count the number of high-confidence keypoints
    total_keypoints = confidence_scores.size  # Total number of keypoints

    # Return video filename and computed confidence percentage
    return os.path.basename(video_path), high_confidence / total_keypoints  # Confidence percentage

# Function to process all videos in a given folder and save results to summary files and individual CSVs
def process_videos(input_folder, output_folder, summary_file):
    if not os.path.exists(input_folder):
        print(f"Folder {input_folder} does not exist.")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the summary file to write confidence scores for each video
    with open(summary_file, mode='w') as scores_file:
        scores_file.write("Video,Confidence Percentage\n")

        # Iterate through all .mp4 files in the folder
        for video_file in os.listdir(input_folder):
            if video_file.endswith(".mp4"):
                video_path = os.path.join(input_folder, video_file)
                print(f"Processing video: {video_file}")

                # Define output CSV path for each video
                output_csv_path = os.path.join(output_folder, f"{video_file}_keypoints.csv")

                # Initialize CSV with headers for the first video
                with open(output_csv_path, mode='w', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    # Write header: video name, followed by x0, y0, score0, x1, y1, score1, ..., x16, y16, score16
                    header = ['Video']
                    for i in range(17):
                        header.extend([f"x{i}", f"y{i}", f"score{i}"])
                    csv_writer.writerow(header)

                # Process the video and calculate confidence percentage
                video_name, confidence_percentage = process_video(video_path, output_csv_path)

                # Write the video confidence score to the summary file
                scores_file.write(f"{video_name}, {confidence_percentage:.2%}\n")

# Define folder paths for fall and no-fall videos
fall_videos_folder = os.path.join("data", "videos", "fall")
no_fall_videos_folder = os.path.join("data", "videos", "no_fall")

# Define output folders for saving the CSV files and the summary files
fall_output_folder = "fall_keypoints"
no_fall_output_folder = "no_fall_keypoints"
fall_summary_file = "fall_scores_summary.txt"
no_fall_summary_file = "no_fall_scores_summary.txt"

# Process fall videos and save the results to individual CSVs and summary file
print("Processing Fall Videos...")
process_videos(fall_videos_folder, fall_output_folder, fall_summary_file)

# Process no-fall videos and save the results to individual CSVs and summary file
print("Processing No Fall Videos...")
process_videos(no_fall_videos_folder, no_fall_output_folder, no_fall_summary_file)

print("The process is complete!")
