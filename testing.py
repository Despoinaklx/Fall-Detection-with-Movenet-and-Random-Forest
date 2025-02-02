import tensorflow as tf
import numpy as np
import cv2
import joblib  # For loading the trained Random Forest model

# Load the TensorFlow Lite Model
interpreter = tf.lite.Interpreter(model_path='3.tflite')
interpreter.allocate_tensors()

# Load the trained Random Forest model
rf_model = joblib.load(r"C:\Users\d2907\file2\fall_detection_model_rf.pkl")  # Change to your actual model file

# Function to draw keypoints on the frame
def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape  # Frame dimensions
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)

# Define keypoint connections
EDGES = {
    (0, 1): 'm', (0, 2): 'c', (1, 3): 'm', (2, 4): 'c',
    (0, 5): 'm', (0, 6): 'c', (5, 7): 'm', (7, 9): 'm',
    (6, 8): 'c', (8, 10): 'c', (5, 6): 'y', (5, 11): 'm',
    (6, 12): 'c', (11, 12): 'y', (11, 13): 'm', (13, 15): 'm',
    (12, 14): 'c', (14, 16): 'c'
}

# Function to draw connections between keypoints
def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

# Function to extract keypoint features with padding
def extract_features(keypoints, confidence_threshold=0.2, min_detected_keypoints=4, expected_features=51):
    shaped = np.squeeze(keypoints)
    valid_keypoints = shaped[shaped[:, 2] > confidence_threshold]
    if len(valid_keypoints) < min_detected_keypoints:
        return None
    feature_vector = shaped[:, :2].flatten()
    
    # Ensure the feature vector has the expected length by padding with zeros if necessary
    if len(feature_vector) < expected_features:
        feature_vector = np.pad(feature_vector, (0, expected_features - len(feature_vector)), 'constant')
    return feature_vector

# Open webcam
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Failed to capture frame")
        continue
    
    img = frame.copy()
    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 256, 256)
    input_img = tf.cast(img, dtype=tf.float32)
    
    # Run inference
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], np.array(input_img))
    interpreter.invoke()
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    
    # Extract features for the classifier
    feature_vector = extract_features(keypoints_with_scores)
    
    if feature_vector is None:
        label = "No Fall (Low Confidence)"
        color = (255, 255, 0)
    else:
        try:
            fall_prediction = rf_model.predict([feature_vector])[0]
            label = "Fall Detected!" if fall_prediction == 1 else "No Fall"
            color = (0, 0, 255) if fall_prediction == 1 else (0, 255, 0)
        except Exception as e:
            print("Prediction Error:", e)
            label = "Prediction Error"
            color = (0, 0, 255)
    
    draw_connections(frame, keypoints_with_scores, EDGES, 0.3)
    draw_keypoints(frame, keypoints_with_scores, 0.3)
    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow('Fall Detection System', frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()