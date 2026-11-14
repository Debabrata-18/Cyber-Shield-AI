
import joblib

print("===== CYBER SHIELD AI TESTING =====")

# Load the upgraded model
model = joblib.load("cyber_model.pkl")

# New test messages
messages = [
    "Good morning, your school class starts at 9 AM.",
    "Your homework submission deadline is tomorrow.",
    "You have received an unexpected prize offer.",
    "An unknown sender has shared a suspicious link.",
    "Send your bank OTP and password to unlock your account.",
    "A fake bank representative is requesting your PIN."
]

# Predict categories
predictions = model.predict(messages)
probabilities = model.predict_proba(messages)

for i, message in enumerate(messages):
    print("\nMessage:", message)
    print("Prediction:", predictions[i])
    print(
        "Model confidence:",
        round(max(probabilities[i]) * 100, 2), "%"
    )

print("\n===== TESTING COMPLETED =====")
