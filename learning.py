import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.impute import SimpleImputer
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data (X_train, X_test, y_train, y_test)
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv")
y_test = pd.read_csv("y_test.csv")

# Load video file mapping
X_test_df = pd.read_csv("test_data.csv")  # Contains 'video_file' column

# Drop non-numeric data
X_train = X_train.select_dtypes(include=[np.number])
X_test = X_test.select_dtypes(include=[np.number])

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Hyperparameter tuning for Random Forest
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf_model = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
grid_search = RandomizedSearchCV(rf_model, param_grid, n_iter=5, cv=3, scoring='accuracy', n_jobs=-1, random_state=42)

grid_search.fit(X_train, y_train.values.ravel())
best_model = grid_search.best_estimator_

# Predict on test set
y_pred = best_model.predict(X_test)
X_test_df["y_pred"] = y_pred  # Add predictions to dataframe

# Majority vote classification per video
video_predictions_majority = X_test_df.groupby("video_file")["y_pred"].agg(lambda x: x.mode()[0])

# Load true video labels
y_test_videos = pd.read_csv("test_videos.csv")

# Assign true labels using majority vote approach
test_data = pd.read_csv("test_data.csv")
video_labels = test_data.groupby("video_file")["label"].agg(lambda x: x.value_counts().idxmax())
y_test_videos["true_label"] = y_test_videos["video_file"].map(video_labels)

# Merge true labels with majority vote predictions
video_results_majority = y_test_videos.merge(video_predictions_majority, on="video_file", how="left")
video_results_majority.rename(columns={"y_pred": "predicted_label"}, inplace=True)

# Evaluate at video level (Majority Vote)
accuracy_majority = accuracy_score(video_results_majority["true_label"], video_results_majority["predicted_label"])
print(f"Accuracy (Video Level - Majority Vote): {accuracy_majority:.4f}")
print("Classification Report (Majority Vote):")
print(classification_report(video_results_majority["true_label"], video_results_majority["predicted_label"]))

# Confusion Matrix (Majority Vote)
cm_majority = confusion_matrix(video_results_majority["true_label"], video_results_majority["predicted_label"])
sns.heatmap(cm_majority, annot=True, fmt="g", cmap="Blues", xticklabels=["No Fall", "Fall"], yticklabels=["No Fall", "Fall"])
plt.title("Confusion Matrix (Video Level - Majority Vote)")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# Save the trained model
joblib.dump(best_model, 'fall_detection_model_rf.pkl')
