from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import torch.nn as nn
from transformers import BartTokenizer, BartForConditionalGeneration

app = Flask(__name__)
CORS(app)


model_name = "../SentenceModels/modelsMSData"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

def summarize_text_bart(input_text):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128)

    outputs = model.generate(**inputs, max_length=50, num_beams=4, early_stopping=True)

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compress', methods=['POST'])
def compress_text():
    data = request.json
    input_text = data.get("text", "")
    compression_level = data.get("compression_level", 1)

    if not input_text:
        return jsonify({"error": "No text provided"}), 400
    if not (1 <= compression_level <= 5):
        return jsonify({"error": "Compression level must be between 1 and 5"}), 400
    for i in range(compression_level):
        input_text = summarize_text_bart(input_text)

    return jsonify({"compressed": input_text})

if __name__ == '__main__':
    app.run(debug=True)
