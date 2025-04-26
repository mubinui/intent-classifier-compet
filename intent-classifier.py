from flask import Flask, request, jsonify
import numpy as np
import onnxruntime
from transformers import AutoTokenizer
import os
import json

app = Flask(__name__)

# Load the model and tokenizer
MODEL_PATH = os.environ.get('MODEL_PATH', './model_onnx')

try:
    print(f"Loading model from {MODEL_PATH}...")
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # Load ONNX model using onnxruntime directly
    onnx_model_path = os.path.join(MODEL_PATH, "model.onnx")
    session = onnxruntime.InferenceSession(onnx_model_path)

    # Load config to get labels
    config_path = os.path.join(MODEL_PATH, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Get intent labels from config
    id_to_intent = config.get('id2label', {})

    # Convert string keys to integers
    id_to_intent = {int(k): v for k, v in id_to_intent.items()}

    # If id2label isn't in the config, create a default mapping
    if not id_to_intent:
        id_to_intent = {
            0: "PRODUCT_DISCOVERY",
            1: "SPECIFIC_PRODUCT",
            2: "ATTRIBUTE_SEARCH",
            3: "PROBLEM_SOLUTION",
            4: "COMPARISON",
            5: "PRICE_BASED",
            6: "AVAILABILITY"
        }

    print(f"Model loaded successfully with ONNX Runtime")
    print(f"Detected intents: {list(id_to_intent.values())}")

except Exception as e:
    print(f"Error loading model: {e}")
    session = None


# Always use your custom mapping
id_to_intent = {
    0: "PRODUCT_DISCOVERY",
    1: "SPECIFIC_PRODUCT",
    2: "ATTRIBUTE_SEARCH",
    3: "PROBLEM_SOLUTION",
    4: "COMPARISON",
    5: "PRICE_BASED",
    6: "AVAILABILITY"
}

@app.route('/')
def home():
    return """
    <h1>E-commerce Intent Classification API</h1>
    <p>Use the /classify endpoint to predict the intent of e-commerce queries.</p>
    <h2>Example:</h2>
    <pre>
    curl -X POST http://localhost:8000/classify \\
        -H "Content-Type: application/json" \\
        -d '{"query": "do you have the new MacBook Pro in stock"}'
    </pre>
    """


# After loading id_to_intent from config.json
# id_to_intent = {int(k): v for k, v in id_to_intent.items()}

@app.route('/ai/classify', methods=['POST'])
def classify():
    if session is None:
        return jsonify({'error': 'Model not loaded. Check server logs for details.'}), 500

    data = request.json
    if not data or 'query' not in data:
        return jsonify({'error': 'Missing query parameter'}), 400

    query = data['query']

    try:
        inputs = tokenizer(
            query,
            return_tensors="np",
            padding='max_length',
            truncation=True,
            max_length=128
        )
        input_feed = {k: inputs[k] for k in [inp.name for inp in session.get_inputs()]}
        outputs = session.run(None, input_feed)
        logits = outputs[0]
        probabilities = softmax(logits)[0]
        prediction = int(np.argmax(logits))
        predicted_intent = id_to_intent.get(prediction+1, f"Unknown Intent ({prediction})")
        return jsonify({
            'intent': predicted_intent
        })
    except Exception as e:
        print(f"Classification error: {e}")
        return jsonify({'error': str(e)}), 500
# Define softmax function
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


@app.route('/ai/health')
def health():
    if session is None:
        return jsonify({'status': 'error', 'message': 'Model not loaded'}), 500
    return jsonify({'status': 'ok', 'message': 'Service is running'})


@app.route('/ai/model_info')
def model_info():
    if session is None:
        return jsonify({'status': 'error', 'message': 'Model not loaded'}), 500

    return jsonify({
        'runtime': 'ONNX Runtime',
        'intents': list(id_to_intent.values()),
        'model_path': MODEL_PATH,
        'tokenizer_vocab_size': len(tokenizer) if tokenizer else None
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)