
import pandas as pd

# Step 1: Load the dataset
df = pd.read_csv("dataset.csv")

print("ORIGINAL DATASET")
print(df)

# Step 2: Check missing values
print("\nMISSING VALUES")
print(df.isnull().sum())

# Step 3: Remove duplicate records
df = df.drop_duplicates()

# Step 4: Remove rows with missing data
df = df.dropna(subset=["message", "label"])

# Step 5: Remove empty messages
df["message"] = df["message"].astype(str).str.strip()
df["label"] = df["label"].astype(str).str.strip()

df = df[df["message"] != ""]
df = df[df["label"] != ""]

# Step 6: Check the cleaned dataset
print("\nCLEANED DATASET")
print(df)

# Step 7: Display dataset information
print("\nDATASET INFORMATION")
print("Total records:", len(df))
print("Categories:")
print(df["label"].value_counts())

# Step 8: Save cleaned dataset
df.to_csv("cleaned_dataset.csv", index=False)

print("\nDataset cleaned and saved successfully!")
