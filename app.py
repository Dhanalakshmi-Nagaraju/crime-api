# app.py
from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Categories and model
crime_categories = {
    "Rape": 10,
    "Murder": 10,
    "Acid Attack": 9,
    "Attempt to Rape": 9,
    "Kidnapping/Abduction": 8,
    "Molestation": 7,
    "Stalking": 6,
    "Harassment": 6,
    "Eve Teasing": 5,
    "Trafficking": 8
}

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    news_text = data.get("text", "")
    CONFIDENCE_THRESHOLD = 0.3
    TOP_N = 3

    if not news_text:
        return jsonify({"error": "Text is required"}), 400

    result = classifier(news_text, list(crime_categories.keys()), multi_label=True)

    filtered = [
        (label, score) for label, score in zip(result["labels"], result["scores"])
        if score >= CONFIDENCE_THRESHOLD
    ]

    filtered = sorted(filtered, key=lambda x: x[1], reverse=True)

    if len(filtered) == 0:
        return jsonify({
            "message": "No strong match found.",
            "assigned_severity_rating": 0
        })

    selected = filtered[:TOP_N]
    output = []

    for label, score in selected:
        output.append({
            "category": label,
            "confidence": round(score, 3),
            "severity": crime_categories[label]
        })

    top_label, top_score = max(selected, key=lambda x: x[1])

    return jsonify({
        "predictions": output,
        "top_category": top_label,
        "top_confidence": round(top_score, 3)
    })

if __name__ == "__main__":
    app.run(debug=True)
