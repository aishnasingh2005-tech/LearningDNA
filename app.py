from flask import Flask, render_template, request, send_file, redirect
import pickle
import pandas as pd
import sqlite3
import csv
import matplotlib
import os
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from datetime import datetime

from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

app = Flask(__name__)

# Create Database

def create_database():

    conn = sqlite3.connect("learning_dna.db")

    cursor = conn.cursor()

    cursor.execute("""

    CREATE TABLE IF NOT EXISTS predictions(

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        name TEXT,

        age INTEGER,
        lessons INTEGER,
        quiz REAL,
        learning_style TEXT,
        confidence REAL,
        created_at TEXT
    )
    """)
    conn.commit()
    conn.close()
create_database()

# Load Model and Encoders

with open("models/learning_dna_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("models/encoders.pkl", "rb") as file:
    encoders = pickle.load(file)

# Store Latest Prediction

latest_report = {}

# Home Page

@app.route("/")
def home():

    conn = sqlite3.connect("learning_dna.db")
    cursor = conn.cursor()

    # Total Predictions
    cursor.execute("SELECT COUNT(*) FROM predictions")
    total_predictions = cursor.fetchone()[0]

    # Average Confidence
    cursor.execute("SELECT AVG(confidence) FROM predictions")
    avg_conf = cursor.fetchone()[0]

    if avg_conf is None:
        avg_conf = 0

    # Count Each Learning Style
    styles = [
        "Visual Learner",
        "Audio Learner",
        "Reading Learner",
        "Practice Learner"
    ]

    counts = {}

    for style in styles:

        cursor.execute(
            "SELECT COUNT(*) FROM predictions WHERE learning_style=?",
            (style,)
        )

        counts[style] = cursor.fetchone()[0]

    conn.close()
    

    return render_template(
        
        
        "index.html",
        total_predictions=total_predictions,
        average_confidence=round(avg_conf, 2),
        visual=counts["Visual Learner"],
        audio=counts["Audio Learner"],
        reading=counts["Reading Learner"],
        practice=counts["Practice Learner"]
    )


# Prediction

@app.route("/predict", methods=["POST"])
def predict():

    global latest_report

    # User Inputs
    name = request.form["name"]

    age = int(request.form["age"])

    lessons_completed = int(request.form["lessons_completed"])

    quiz_avg_score = float(request.form["quiz_avg_score"])

    learning_method = request.form["learning_method"]

    # Convert Learning Method
    
    if learning_method == "Watching Videos":
        video_preference = "High"

    elif learning_method == "Reading Notes":
        video_preference = "Medium"

    elif learning_method == "Listening":
        video_preference = "Low"

    else:
        video_preference = "Medium"

    # Encode Feature

    video_encoded = encoders["video_preference"].transform(
        [video_preference]
    )[0]

    # Create DataFrame

    sample = pd.DataFrame([{

        "age": age,

        "lessons_completed": lessons_completed,

        "quiz_avg_score": quiz_avg_score,

        "video_preference": video_encoded

    }])

    # Prediction

    prediction = model.predict(sample)[0]

    confidence = model.predict_proba(sample).max() * 100

    learning_style = encoders["learning_style"].inverse_transform(
        [prediction]
    )[0]

    # Learning Style Details

    if learning_style == "Visual Learner":

        recommendations = [

            "📺 Watch YouTube Videos",

            "🧠 Use Mind Maps",

            "📊 Study using Flowcharts",

            "🎨 Use Infographics"

        ]

        strengths = [

            "Excellent visual memory",

            "Learns quickly through diagrams",

            "Understands flowcharts easily"

        ]

        weaknesses = [

            "Can lose focus during long lectures",

            "Needs visual content",

            "Practice active recall"

        ]

    elif learning_style == "Audio Learner":

        recommendations = [

            "🎧 Listen to Podcasts",

            "🎤 Revise by Speaking",

            "🎵 Record Audio Notes",

            "👥 Join Group Discussions"

        ]

        strengths = [

            "Excellent listening skills",

            "Learns well from discussions",

            "Strong verbal communication"

        ]

        weaknesses = [

            "Can forget written content",

            "Needs quiet surroundings",

            "Should make written notes"

        ]

    elif learning_style == "Reading Learner":

        recommendations = [

            "📖 Read Books",

            "📝 Make Written Notes",

            "📚 Study PDFs",

            "✍ Practice Revision"

        ]

        strengths = [

            "Excellent reading comprehension",

            "Makes organized notes",

            "Strong theoretical understanding"

        ]

        weaknesses = [

            "May read slowly",

            "Needs regular revision",

            "Should include visual learning"

        ]

    else:

        recommendations = [

            "💻 Build Mini Projects",

            "🛠 Hands-on Practice",

            "🧪 Experiment Yourself",

            "🎯 Solve Real Problems"

        ]

        strengths = [

            "Excellent practical skills",

            "Learns by doing",

            "Strong problem-solving ability"

        ]

        weaknesses = [

            "Can ignore theory",

            "Needs concept revision",

            "Should improve documentation"

        ]

    # Save Report Data

    latest_report = {

        "name": name,

        "prediction": learning_style,

        "confidence": round(confidence, 2),

        "recommendations": recommendations,

        "strengths": strengths,

        "weaknesses": weaknesses

    }
    # Save Prediction to Database

    conn = sqlite3.connect("learning_dna.db")

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO predictions
        (
            name,
            age,
            lessons,
            quiz,
            learning_style,
            confidence,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            name,
            age,
            lessons_completed,
            quiz_avg_score,
            learning_style,
            round(confidence, 2),
            datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        )
    )

    conn.commit()
    conn.close()

    return render_template(

        "result.html",

        name=name,

        prediction=learning_style,

        confidence=round(confidence, 2),

        recommendations=recommendations,

        strengths=strengths,

        weaknesses=weaknesses

    )

# Prediction History

@app.route("/history")
def history():

    conn = sqlite3.connect("learning_dna.db")

    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            id,
            name,
            age,
            learning_style,
            confidence,
            created_at
        FROM predictions
        ORDER BY id DESC
    """)

    records = cursor.fetchall()

    conn.close()

    return render_template(
        "history.html",
        records=records
    )

# Delete Prediction
@app.route("/delete/<int:id>")
def delete(id):

    conn = sqlite3.connect("learning_dna.db")

    cursor = conn.cursor()

    cursor.execute(
        "DELETE FROM predictions WHERE id=?",
        (id,)
    )

    conn.commit()

    conn.close()

    return redirect("/history")

# Export Prediction History to CSV
@app.route("/export")
def export():

    conn = sqlite3.connect("learning_dna.db")

    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            id,
            name,
            age,
            lessons,
            quiz,
            learning_style,
            confidence,
            created_at
        FROM predictions
        ORDER BY id DESC
    """)

    rows = cursor.fetchall()

    conn.close()

    csv_file = "Prediction_History.csv"

    with open(csv_file, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "ID",
            "Name",
            "Age",
            "Lessons",
            "Quiz Score",
            "Learning Style",
            "Confidence",
            "Date"
        ])

        writer.writerows(rows)

    return send_file(
        csv_file,
        as_attachment=True
    )

# Download PDF Report

@app.route("/download")
def download():

    global latest_report

    pdf_file = "Learning_DNA_Report.pdf"

    doc = SimpleDocTemplate(pdf_file)

    styles = getSampleStyleSheet()

    story = []

    # ==========================
    # Title
    # ==========================

    story.append(
        Paragraph(
            "<b>Learning DNA Engine Report</b>",
            styles["Title"]
        )
    )

    story.append(
        Paragraph("<br/>", styles["BodyText"])
    )

    # ==========================
    # Student Details
    # ==========================

    story.append(
        Paragraph(
            f"<b>Student Name :</b> {latest_report.get('name','N/A')}",
            styles["Heading2"]
        )
    )

    story.append(
        Paragraph(
            f"<b>Learning Style :</b> {latest_report.get('prediction','N/A')}",
            styles["Heading2"]
        )
    )

    story.append(
        Paragraph(
            f"<b>Confidence :</b> {latest_report.get('confidence',0)} %",
            styles["Heading2"]
        )
    )

    story.append(
        Paragraph("<br/>", styles["BodyText"])
    )

    # ==========================
    # Recommendations
    # ==========================

    story.append(
        Paragraph(
            "<b>Recommended Study Methods</b>",
            styles["Heading2"]
        )
    )

    for item in latest_report.get("recommendations", []):

        story.append(
            Paragraph(
                "• " + item,
                styles["BodyText"]
            )
        )

    story.append(
        Paragraph("<br/>", styles["BodyText"])
    )

    # ==========================
    # Strengths
    # ==========================

    story.append(
        Paragraph(
            "<b>Strengths</b>",
            styles["Heading2"]
        )
    )

    for item in latest_report.get("strengths", []):

        story.append(
            Paragraph(
                "• " + item,
                styles["BodyText"]
            )
        )

    story.append(
        Paragraph("<br/>", styles["BodyText"])
    )

    # Areas to Improve
    story.append(
        Paragraph(
            "<b>Areas to Improve</b>",
            styles["Heading2"]
        )
    )

    for item in latest_report.get("weaknesses", []):

        story.append(
            Paragraph(
                "• " + item,
                styles["BodyText"]
            )
        )

    story.append(
        Paragraph("<br/>", styles["BodyText"])
    )

    # Weekly Study Plan
    story.append(
        Paragraph(
            "<b>Weekly AI Study Plan</b>",
            styles["Heading2"]
        )
    )

    weekly_plan = [
        "Monday - Watch educational videos",
        "Tuesday - Create mind maps",
        "Wednesday - Attempt quizzes",
        "Thursday - Practice hands-on exercises",
        "Friday - Revise all topics"
    ]

    for day in weekly_plan:

        story.append(
            Paragraph(
                "• " + day,
                styles["BodyText"]
            )
        )

    story.append(
        Paragraph("<br/>", styles["BodyText"])
    )

    # Footer
    story.append(
        Paragraph(
            "<i>Generated by Learning DNA Engine using Flask + Machine Learning</i>",
            styles["Italic"]
        )
    )

    doc.build(story)

    return send_file(
        pdf_file,
        as_attachment=True
    )

# Run Flask
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)