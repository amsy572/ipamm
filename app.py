import os
import requests
import logging
import json
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from time import sleep

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# GitHub repository URL for the model files
model_repo_url = "https://github.com/amsy572/ipamm/raw/main/fine_tuned_hajj_qa_model"
model_dir = "/tmp/fine_tuned_hajj_qa_model"  # Path to save downloaded model files

# Ensure the model directory exists
os.makedirs(model_dir, exist_ok=True)

# List of expected model files and their raw URLs
model_files = [
    ("config.json", f"{model_repo_url}/config.json"),
    ("model.safetensors", f"{model_repo_url}/model.safetensors"),
    ("tokenizer_config.json", f"{model_repo_url}/tokenizer_config.json"),
    ("special_tokens_map.json", f"{model_repo_url}/special_tokens_map.json"),
    ("training_args.bin", f"{model_repo_url}/training_args.bin"),
    ("vocab.txt", f"{model_repo_url}/vocab.txt"),
    ("context.json", f"{model_repo_url}/context.json")
]

# Download the model files with retry logic
for file_name, file_url in model_files:
    success = False
    for attempt in range(3):  # Retry up to 3 times
        response = requests.get(file_url)
        if response.status_code == 200:
            with open(os.path.join(model_dir, file_name), 'wb') as f:
                f.write(response.content)
            logger.info(f"Downloaded {file_name} successfully.")
            success = True
            break
        else:
            logger.error(f"Failed to download {file_name} from {file_url}. Status code: {response.status_code}")
            logger.debug(response.text)
            sleep(2)  # Wait before retrying
    
    if not success:
        logger.error(f"Failed to download {file_name} after 3 attempts.")
        exit(1)  # Exit if any model file fails to download

# Load the tokenizer and model with error handling
try:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForQuestionAnswering.from_pretrained(model_dir)
    logger.info("Model and tokenizer loaded successfully.")
except Exception as e:
    logger.error("Error loading model or tokenizer:", exc_info=True)
    exit(1)  # Exit the application if loading fails

# Initialize QA pipeline
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Load all contexts from the JSON file
context_file_path = "/tmp/fine_tuned_hajj_qa_model/context.json"  # Path to your context JSON file
try:
    with open(context_file_path, 'r', encoding='utf-8') as f:
        contexts = json.load(f)  # Load all contexts as a list
    logger.info("Loaded contexts successfully.")
except FileNotFoundError:
    logger.error(f"Context file {context_file_path} not found.")
    contexts = []

@app.route('/answer', methods=['POST'])
def get_answer():
    data = request.json
    question = data.get("question", "")
    context_index = data.get("context_index", 0)  # Optionally pass which context to use

    # Validate input
    if not question:
        return jsonify({"error": "Question is required"}), 400  # Return 400 Bad Request
    if context_index >= len(contexts) or context_index < 0:
        return jsonify({"error": "Invalid context index"}), 400  # Return 400 Bad Request

    # Use the specified context or combine contexts if needed
    selected_context = contexts[context_index]['context']  # Example: use one context
    # Or combine all contexts: selected_context = " ".join([ctx['context'] for ctx in contexts])

    try:
        # Get the answer using the QA pipeline
        result = qa_pipeline(question=question, context=selected_context)
        return jsonify({
            "question": question,
            "context": selected_context,
            "answer": result['answer'],
            "score": result['score']
        })
    except Exception as e:
        logger.error("Error during question answering:", exc_info=True)
        return jsonify({"error": "An error occurred while processing the request"}), 500  # Return 500 Internal Server Error

if __name__ == '__main__':
    # Run the Flask app
    app.run(host="0.0.0.0", port=5000, debug=True)  # Use debug=True for local development only
