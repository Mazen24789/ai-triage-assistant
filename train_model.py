import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

print("Loading data...")

encounters = pd.read_csv("encounters.csv")
vitals = pd.read_csv("vitals.csv")

print("Merging data...")

df = encounters.merge(vitals, on=["patient_id", "encounter_id"])

print("Columns:", df.columns)

df = df[[
    "heart_rate",
    "temperature_celsius",
    "o2_saturation",
    "triage_level"
]].dropna()

print("Training model...")

X = df[["heart_rate", "temperature_celsius", "o2_saturation"]]
y = df["triage_level"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("Model accuracy:", accuracy)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ model.pkl saved")