from flask import Flask, request, render_template
import torch
import numpy as np
import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.preprocessing import LabelEncoder

# === Load product data ===
product_df = pd.read_csv("Sample - Superstore.csv")

# === Load tokenizer and model ===
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("checkpoint-900", local_files_only=True)
model.eval()

# === Load label encoder ===
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['compare_prices', 'get_cheapest', 'other'])

def extract_keyword(text):
    stopwords = {
        "all", "the", "show", "me", "please", "compare", "get", "cheapest",
        "for", "price", "prices", "of", "a", "an", "find", "list"
    }
    words = [word.lower() for word in text.split() if word.lower() not in stopwords]
    product_names = product_df["Product Name"].str.lower()

    best_word = None
    most_matches = 0

    for word in words:
        match_count = product_names.str.contains(rf"\b{word}\b", na=False).sum()
        if match_count > most_matches:
            most_matches = match_count
            best_word = word

    return best_word

# === Get cheapest product ===
def get_cheapest_product(keyword):
    if not keyword:
        return "❌ Sorry, I couldn't find a product keyword in your message."

    filtered = product_df[product_df["Product Name"].str.contains(rf"\b{keyword}\b", case=False, na=False)]

    if filtered.empty:
        return f"❌ Sorry, I couldn't find any products matching: '{keyword}'"

    cheapest = filtered.sort_values("Sales", ascending=True).iloc[0]
    return (
        f"<strong>Cheapest {keyword}:</strong><br>"
        f"- Product: {cheapest['Product Name']}<br>"
        f"- Price: ${cheapest['Sales']:.2f}<br>"
        f"- Location: {cheapest['City']}, {cheapest['State']}"
    )

# === Compare prices (list all matches) ===
def compare_prices(keyword):
    if not keyword:
        return "❌ Sorry, I couldn't find a product keyword in your message."

    filtered = product_df[product_df["Product Name"].str.contains(rf"\b{keyword}\b", case=False, na=False)]

    if filtered.empty:
        return f"❌ Sorry, I couldn't find any products matching: '{keyword}'"

    results = [
        f"{row['Product Name']} – ${row['Sales']:.2f} ({row['City']}, {row['State']})"
        for _, row in filtered.iterrows()
    ]

    return f"<strong>Found {len(results)} match(es) for '{keyword}':</strong><br>" + "<br>".join(results)

# === Predict intent + extract keyword ===
def predict_intent(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_class_id = int(torch.argmax(outputs.logits, dim=1).item())
    predicted_label = label_encoder.inverse_transform([predicted_class_id])[0]
    return predicted_label

# === Flask app ===
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    chat_history = []

    if request.method == "POST":
        action = request.form.get("action")

        if action == "Clear":
            return render_template("index.html", chat_history=[])

        # Restore chat history
        previous_texts = request.form.getlist("history_text")
        previous_responses = request.form.getlist("history_response")
        chat_history = [{"user": t, "response": r} for t, r in zip(previous_texts, previous_responses)]

        user_input = request.form["message"]
        intent = predict_intent(user_input)
        keyword = extract_keyword(user_input)  # ✅ now THIS determines the keyword

        # Respond based on intent
        if intent == "get_cheapest":
            result = get_cheapest_product(keyword)
        elif intent == "compare_prices":
            result = compare_prices(keyword)
        else:
            result = "🤖 I couldn't understand your request. Try asking about a product or price."

        chat_history.append({"user": user_input, "response": result})
        return render_template("index.html", chat_history=chat_history)

    return render_template("index.html", chat_history=[])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000,debug=True)