
import pandas as pd

# Load existing dataset
df = pd.read_csv("cleaned_dataset.csv")

# 15 Safe messages
safe = [
    "Good morning, have a wonderful day.",
    "Your school timetable has been updated.",
    "Please submit your assignment tomorrow.",
    "Your online order has been delivered.",
    "Your meeting is scheduled for Monday.",
    "Thank you for contacting customer support.",
    "Your library book is due next week.",
    "The examination results are now available.",
    "Your payment was successfully completed.",
    "Please attend the scheduled online class.",
    "Your appointment has been confirmed.",
    "The weather report is available today.",
    "Your registration was successful.",
    "Your parcel is ready for collection.",
    "Your password was successfully changed."
]

# 15 Suspicious messages
suspicious = [
    "You have been selected for a special reward.",
    "Claim your exclusive offer before it expires.",
    "Your account requires an urgent security review.",
    "A prize is waiting for you to claim.",
    "You may qualify for a special cash reward.",
    "Your subscription may expire soon, check your account.",
    "An unknown device attempted to access your account.",
    "You have received an unexpected promotional offer.",
    "Verify the unexpected notification through the official website.",
    "Your account activity needs your attention.",
    "You are eligible for a limited time promotion.",
    "An unknown sender has shared an unexpected link.",
    "Your delivery details may need confirmation.",
    "A message claims you have won a competition.",
    "Your account has received an unusual login notification."
]

# 15 Malicious messages
malicious = [
    "Your bank account is blocked, send your password to restore it.",
    "Share your banking OTP with us to receive your refund.",
    "Your account will be closed unless you reveal your login password.",
    "Send your card PIN to claim this fake payment refund.",
    "A scammer is requesting your private banking credentials.",
    "Enter your password on this unofficial banking page.",
    "Share your verification code with an unknown sender.",
    "This message impersonates your bank and requests your PIN.",
    "A fake support agent asks for your account password.",
    "The sender demands confidential login information.",
    "A fraudulent message requests your banking security code.",
    "An unknown person asks you to transfer money secretly.",
    "A fake lottery message demands an advance payment.",
    "The sender threatens account closure to steal credentials.",
    "A fraudulent website requests sensitive personal information."
]

# Create additional dataset
new_data = pd.DataFrame({
    "message": safe + suspicious + malicious,
    "label": (
        ["Safe"] * 15 +
        ["Suspicious"] * 15 +
        ["Malicious"] * 15
    )
})

# Combine with existing dataset
updated_df = pd.concat(
    [df, new_data],
    ignore_index=True
)

# Remove accidental duplicates
updated_df = updated_df.drop_duplicates(
    subset=["message"]
)

# Save updated dataset
updated_df.to_csv(
    "cleaned_dataset.csv",
    index=False
)

print("===== DATASET UPDATED =====")
print("Total messages:", len(updated_df))
print("\nCategory distribution:")
print(updated_df["label"].value_counts())

print("\nDataset saved successfully!")
