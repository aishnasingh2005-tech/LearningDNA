import pandas as pd
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ===========================
# Load Dataset
# ===========================

df = pd.read_csv("dataset/learnkins_students_v3.csv")

# Remove unnecessary columns
df = df.drop(columns=[
    "student_id",
    "full_name",
    "email",
    "phone",
    "referral_code"
])

# ===========================
# Create Learning Style
# ===========================

def assign_learning_style(row):

    if (
        row["quiz_avg_score"] >= 85
        and row["lessons_completed"] >= 50
        and row["streak_days"] >= 20
    ):
        return "Practice Learner"

    elif (
        row["quiz_avg_score"] >= 80
        and row["lessons_completed"] >= 35
    ):
        return "Reading Learner"

    elif row["xp_points"] >= 2500:
        return "Visual Learner"

    else:
        return "Audio Learner"


df["learning_style"] = df.apply(assign_learning_style, axis=1)

# ===========================
# Create Video Preference
# ===========================

video_pref = []

for style in df["learning_style"]:

    if style == "Visual Learner":
        video_pref.append("High")

    elif style == "Audio Learner":
        video_pref.append("Low")

    elif style == "Reading Learner":
        video_pref.append("Medium")

    else:
        video_pref.append("Medium")

df["video_preference"] = video_pref

# ===========================
# Keep Only Required Columns
# ===========================

df = df[[
    "age",
    "lessons_completed",
    "quiz_avg_score",
    "video_preference",
    "learning_style"
]]

# ===========================
# Encode Categorical Columns
# ===========================

encoders = {}

for col in ["video_preference", "learning_style"]:

    encoder = LabelEncoder()

    df[col] = encoder.fit_transform(df[col])

    encoders[col] = encoder

# ===========================
# Split Data
# ===========================

X = df.drop("learning_style", axis=1)

y = df["learning_style"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ===========================
# Train Model
# ===========================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ===========================
# Evaluate
# ===========================

prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

print("Accuracy :", accuracy)

# ===========================
# Save Model
# ===========================

with open("models/learning_dna_model.pkl", "wb") as file:
    pickle.dump(model, file)

with open("models/encoders.pkl", "wb") as file:
    pickle.dump(encoders, file)

print("Model Saved Successfully!")