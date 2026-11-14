
import pandas as pd

df = pd.read_csv("cleaned_dataset.csv")

print("===== CYBER SHIELD AI DATASET REPORT =====")

print("\nTotal messages:", len(df))

print("\nColumns:")
print(df.columns.tolist())

print("\nCategory distribution:")
print(df["label"].value_counts())

print("\nMissing values:")
print(df[["message", "label"]].isnull().sum())

print("\nDuplicate messages:",
      df["message"].duplicated().sum())

print("\nEmpty messages:",
      df["message"].astype(str).str.strip().eq("").sum())

print("\n===== REPORT COMPLETED =====")
