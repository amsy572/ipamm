from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import os

app = Flask(__name__)

# Define the model path (adjust this to the correct path in your deployment)
model_path = os.getenv("MODEL_PATH", "fine_tuned_hajj_qa_model")  # Use environment variable or default

# Load the tokenizer and model with error handling
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
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
