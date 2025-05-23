from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load model
model = joblib.load("sentiment_model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

# Root route to check if server is live
@app.route("/")
def home():
    return "Server is running! Send POST requests to /predict."

# Predict route
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data["text"]
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    return jsonify({"sentiment": prediction})

if __name__ == "__main__":
    app.run(debug=True)