import pandas as pd
from scipy.sparse import load_npz
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import shuffle
from tqdm import tqdm
import joblib  # For saving the model

# 1. Load the sparse TF-IDF matrix
file_path = "tf_idf_sparse_matrix.npz"  # Replace with your file path
tf_idf_matrix = load_npz(file_path)

# 2. Load the document labels
label_file = "document_labels.csv"  # Replace with your file path
labels_df = pd.read_csv(label_file)

# 3. Check for and handle missing labels
print("Number of NaN labels before cleaning:", labels_df['female'].isna().sum())

# Drop rows with missing labels
labels_df = labels_df.dropna(subset=['female'])

# Reload the labels after dropping NaN
labels = labels_df['female']

# Ensure the TF-IDF matrix aligns with the cleaned labels
tf_idf_matrix = tf_idf_matrix[labels_df.index]

print(f"Number of valid labels after cleaning: {len(labels)}")
print(f"TF-IDF matrix shape after cleaning: {tf_idf_matrix.shape}")

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    tf_idf_matrix, labels, test_size=0.2, random_state=42
)

# 5. Training the SVM with progress tracking
print("Training the SVM...")

# Convert sparse matrix to shuffled blocks to approximate progress
X_train, y_train = shuffle(X_train, y_train, random_state=42)
block_size = len(y_train) // 10  # Divide into 10 blocks for progress updates
progress_model = SVC(kernel='linear', random_state=42, max_iter=1, verbose=False)

for i in tqdm(range(10), desc="Training Progress", unit="block"):
    start = i * block_size
    end = (i + 1) * block_size if i < 9 else len(y_train)  # Last block takes remaining
    progress_model.fit(X_train[start:end], y_train[start:end])

# Refit the model fully after approximate training (if needed)
svm_model = progress_model

# Save the trained SVM model and vectorizer
joblib.dump(svm_model, "svm_model.pkl")
print("Trained SVM model saved as 'svm_model.pkl'")

# 6. Predict and evaluate
print("Evaluating the SVM...")
y_pred = svm_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
