
# Fall Detection with MoveNet and Random Forest

This project aims to detect falls using the pre-trained MoveNet model for feature extraction and a Random Forest algorithm for data classification.

---

## 🎯 Objective
The project develops a fall detection system, achieving 85% accuracy. Although this accuracy is promising, the system serves as a foundational step toward a more comprehensive solution. It excels in minimizing false positives but requires further refinement for accurate fall detection.

---

## 📂 Project Structure

### 🔹 Data
- **Fall Dataset**: [Google Drive Link](https://drive.google.com/drive/folders/1sXZ3oKdmdfrku2IpATMXC5SW3qEgaUE7?usp=drive_link)
- **No Fall Dataset**: [Google Drive Link](https://drive.google.com/drive/folders/1LXS3OBtgv3RFs3o7Br5TLXJOw4PAtv5y?usp=drive_link)
- **Annotations**: Labels are stored in `data/csv`.

### 🛠️ Data Processing
- **`extract_keypoints.py`**: 
  - Extracts 17 keypoints (x, y, confidence score) from each video frame.
  - Saves extracted data in CSV format.
  - Output directories:
    - `data/keypoints/fall_keypoints`
    - `data/keypoints/no_fall_keypoints`
- **`cleaned_keypoints.py`**: 
  - Preprocesses data by replacing coordinates with a confidence score < 0.3 using linear interpolation.
  - Stores processed data in `data/cleaned_keypoints`.

### 🔄 Dataset Preparation
- **`preparing_data.py`**: 
  - Groups frames corresponding to each video.
  - Splits data into training (80%) and testing (20%).
  - Generates the following files:
    - `training_dataset.csv` (all frames with labels)
    - `test_videos.csv` & `test_train.csv` (separated videos)
    - `X_train`, `X_test`, `y_train`, `y_test`

### 🧠 Training & Evaluation
- **`learning.py`**: 
  - Loads data and trains the Random Forest model.
  - Uses majority voting to classify entire videos instead of individual frames.
  - Produces **Classification Report** and **Confusion Matrix**.

### 🏗️ Testing & Fall Detection
- **`testing.py`**: 
  - Utilizes MoveNet for real-time feature extraction.
  - Uses the trained model for fall detection.
  - Applies thresholding and requires a minimum number of detected keypoints.

---

## 📌 Conclusion
This project provides a functional pipeline for fall detection,machine learning for classification. While there is room for improvement, its low false positive rate is a significant advantage.

---

💡 **Future Enhancements**:
- Implement advanced algorithms (e.g., LSTMs for temporal sequence analysis).
- Increase dataset diversity.
- Improve accuracy through enhanced feature engineering.
```

