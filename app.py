import os
import requests
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

app = Flask(__name__)

# Define the GitHub raw URL for the model files
model_url = "https://github.com/amsy572/ipamm/raw/main/fine_tuned_hajj_qa_model"
model_dir = "/tmp/fine_tuned_hajj_qa_model"  # Path to save downloaded model files

# Ensure the model directory exists
os.makedirs(model_dir, exist_ok=True)

# List of expected model files and their raw URLs
model_files = [
    ("config.json", f"{model_url}/config.json"),
    ("pytorch_model.bin", f"{model_url}/pytorch_model.bin"),
    ("tokenizer_config.json", f"{model_url}/tokenizer_config.json"),
    ("vocab.txt", f"{model_url}/vocab.txt")
]

# Download the model files from GitHub
for file_name, file_url in model_files:
    response = requests.get(file_url)
    if response.status_code == 200:
        with open(os.path.join(model_dir, file_name), 'wb') as f:
            f.write(response.content)
    else:
        print(f"Failed to download {file_name} from {file_url}")
        exit(1)  # Exit if any model file fails to download

# Load the tokenizer and model with error handling
try:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForQuestionAnswering.from_pretrained(model_dir)
except Exception as e:
    print("Error loading model or tokenizer:", e)
    exit(1)  # Exit the application if loading fails

# Initialize QA pipeline
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

@app.route('/answer', methods=['POST'])
def get_answer():
    data = request.json
    question = data.get("question", "")
    context = "Jamarat suna da duwatsu uku a Mina waɗanda ke wakiltar Shaidan. Alhazai suna jifansu da duwatsu yayin Hajj."

    if not question or not context:
        return jsonify({"error": "Both question and context are required"}), 400

    try:
        # Get the answer using the QA pipeline
        result = qa_pipeline(question=question, context=context)
        return jsonify({
            "answer": result['answer'],
            "score": result['score']
        })
    except Exception as e:
        print("Error during question answering:", e)
        return jsonify({"error": "An error occurred while processing the request"}), 500

if __name__ == '__main__':
    # Run the Flask app
    app.run(host="0.0.0.0", port=5000, debug=True)  # Use debug=True for local development only
