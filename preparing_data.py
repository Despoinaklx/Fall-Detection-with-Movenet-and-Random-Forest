import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Ρυθμίσεις
base_dir = "data/cleaned_keypoints"  # Φάκελος που περιέχει τα καθαρισμένα keypoints
output_file = "data/split/training_dataset.csv"  # Όνομα αρχείου για το τελικό dataset

# Λίστα φακέλων με ετικέτες (1 για πτώση, 0 για μη πτώση)
folders = [
    ("data/cleaned_keypoints/cleaned_fall_keypoints", 1),  # Πτώση
    ("data/cleaned_keypoints/cleaned_no_fall_keypoints", 0),  # Μη πτώση
]

def create_training_dataset(folders):
    # Αρχικοποίηση λίστας για αποθήκευση δεδομένων
    all_data = []
    video_labels = {}  # Λεξικό για την παρακολούθηση των ετικετών ανά βίντεο
    expected_columns = None  # Χώρος για έλεγχο συνέπειας των στηλών
    
    # Επεξεργασία κάθε φακέλου (για πτώση και μη πτώση δεδομένα)
    for folder, label in folders:
        # Επεξεργασία κάθε αρχείου στο φάκελο
        for file in os.listdir(folder):
            if file.endswith(".csv"):  # Επεξεργασία μόνο των αρχείων .csv
                file_path = os.path.join(folder, file)
                
                # Παράβλεψη άδειων αρχείων
                if os.stat(file_path).st_size == 0:
                    print(f"Skipping empty file: {file_path}")
                    continue
                
                # Φόρτωση του αρχείου CSV σε DataFrame
                data = pd.read_csv(file_path)
                
                # Αρχικοποίηση των αναμενόμενων στηλών αν είναι το πρώτο αρχείο
                if expected_columns is None:
                    expected_columns = set(data.columns)
                
                # Έλεγχος για συνέπεια στις στήλες
                if set(data.columns) != expected_columns:
                    print(f"Skipping file due to column mismatch: {file_path}")
                    continue
                
                # Προσθήκη στήλης 'label' για την ένδειξη αν είναι πτώση (1) ή μη πτώση (0)
                data['label'] = label
                data['video_file'] = os.path.join(folder, file)  # Προσθήκη του πλήρους μονοπατιού του αρχείου
                # Προσθήκη των δεδομένων στην λίστα
                all_data.append(data)
                # Παρακολούθηση της ετικέτας για το βίντεο (πλήρες μονοπάτι)
                video_labels[file_path] = label

    # Συγχώνευση όλων των δεδομένων σε ένα ενιαίο DataFrame
    training_data = pd.concat(all_data, ignore_index=True)
    return training_data, video_labels

# Δημιουργία του dataset
training_data, video_labels = create_training_dataset(folders)

# Αποθήκευση του τελικού dataset
training_data.to_csv(output_file, index=False)
print(f"Dataset saved to {output_file}")

# Εξαγωγή μοναδικών βίντεο (πλήρη μονοπάτια)
unique_videos = [os.path.join(folder, file) for folder, _ in folders for file in os.listdir(folder) if file.endswith(".csv")]
video_labels_list = [video_labels[v] for v in unique_videos if v in video_labels]

# Διαχωρισμός σε Train και Test sets (stratified με βάση την ετικέτα)
train_videos, test_videos = train_test_split(
    unique_videos, test_size=0.2, random_state=42, stratify=video_labels_list
)

# Διασφάλιση ότι δεν υπάρχει επικάλυψη μεταξύ των train και test sets
assert set(train_videos).isdisjoint(test_videos), "Train and Test sets contain overlapping videos!"

print("No overlapping videos found.")

# Φιλτράρισμα των δεδομένων ανάλογα με το όνομα του βίντεο (για να διασφαλιστεί ότι δεν υπάρχει επικάλυψη)
train_data = training_data[training_data['video_file'].isin(train_videos)]
test_data = training_data[training_data['video_file'].isin(test_videos)]

# Αποθήκευση των δεδομένων για train και test
train_data.to_csv("train_data.csv", index=False)
test_data.to_csv("test_data.csv", index=False)

# Προετοιμασία των χαρακτηριστικών (X) και ετικετών (y)
X_train = train_data.drop(columns=["label", "video_file"])  # Χαρακτηριστικά για εκπαίδευση
y_train = train_data["label"]  # Ετικέτες για εκπαίδευση
X_test = test_data.drop(columns=["label", "video_file"])  # Χαρακτηριστικά για τεστ
y_test = test_data["label"]  # Ετικέτες για τεστ

# Αποθήκευση των Train και Test datasets
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

# Αποθήκευση λιστών με τα βίντεο για Train/Test
pd.DataFrame(train_videos, columns=["video_file"]).to_csv("train_videos.csv", index=False)
pd.DataFrame(test_videos, columns=["video_file"]).to_csv("test_videos.csv", index=False)

print("Training and test sets saved.")
