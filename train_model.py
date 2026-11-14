
import pandas as pd
import joblib
import os
import shutil

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

print("===== CYBER SHIELD AI TRAINING =====")

# 1. Load dataset
df = pd.read_csv("cleaned_dataset.csv")

# 2. Clean and validate
df = df.dropna(subset=["message", "label"]).copy()

df["message"] = df["message"].astype(str).str.strip()
df["label"] = df["label"].astype(str).str.strip().str.title()

df = df[df["message"] != ""]
df = df.drop_duplicates(subset=["message"])

allowed_labels = {"Safe", "Suspicious", "Malicious"}

if set(df["label"].unique()) != allowed_labels:
    raise ValueError("Dataset must contain all three valid categories.")

if len(df) < 10:
    raise ValueError("Not enough training messages.")

print("\nDataset size:", len(df))
print("\nCategories:")
print(df["label"].value_counts())

# 3. Split training and testing data
X = df["message"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("\nTraining messages:", len(X_train))
print("Testing messages:", len(X_test))

# 4. Create improved ML pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True
    )),

    ("classifier", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ))
])

# 5. Train model
print("\nTraining AI model...")
model.fit(X_train, y_train)

# 6. Evaluate model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n===== MODEL PERFORMANCE =====")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    labels=["Safe", "Suspicious", "Malicious"],
    zero_division=0
))

# 7. Train final model on all messages
print("\nTraining final model on all 100 messages...")
model.fit(X, y)

# 8. Save model safely
temporary_file = "cyber_model_new.pkl"
backup_file = "cyber_model_backup.pkl"

try:
    joblib.dump(model, temporary_file)

    # Verify saved model
    verified_model = joblib.load(temporary_file)

    test_prediction = verified_model.predict(
        ["Your account needs verification."]
    )

    print("\nModel verification successful!")
    print("Test prediction:", test_prediction[0])

    # Preserve previous model
    if os.path.exists("cyber_model.pkl"):
        shutil.copy2("cyber_model.pkl", backup_file)
        print("Previous model backed up.")

    # Replace old model
    os.replace(temporary_file, "cyber_model.pkl")

    print("\nSUCCESS!")
    print("New model saved as cyber_model.pkl")
    print("Categories:", verified_model.classes_)

except Exception as error:
    print("\nTraining or saving failed:", error)

    if os.path.exists(temporary_file):
        os.remove(temporary_file)

print("\n===== TRAINING COMPLETED =====")
