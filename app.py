# app.py

import base64
from io import BytesIO
from typing import Tuple

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import torch
from torchvision import transforms
from flask import Flask, jsonify, render_template, request

import config
from model import MnistModel  # Import from the new model.py file

# --- App Initialization ---
app = Flask(__name__)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# --- Load Model (GLOBAL SCOPE) ---
# This runs once when the app starts, making the model available to all requests.
MODEL = MnistModel(classes=10)
try:
    MODEL.load_state_dict(torch.load('checkpoint/mnist.pt', map_location=DEVICE))
    print('Loaded model from checkpoint/mnist.pt')
except FileNotFoundError:
    print('ERROR: Model checkpoint not found. Run train.py to create it.')
MODEL.to(DEVICE)
MODEL.eval()


def figure_to_base64(figure: plt.Figure) -> str:
    """Convert a Matplotlib figure to a base64 PNG string."""
    buffer = BytesIO()
    figure.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(figure)
    return encoded


def probability_plot_image(probabilities: list) -> str:
    """Create a bar plot of class probabilities (0-9) and return base64 PNG."""
    figure, axis = plt.subplots(figsize=(6, 3))
    classes = list(range(10))
    axis.bar(classes, probabilities, color='#4C78A8')
    axis.set_xticks(classes)
    axis.set_xlabel('Digit')
    axis.set_ylabel('Probability (%)')
    axis.set_ylim(0, 100)
    axis.grid(True, axis='y', linestyle='--', alpha=0.3)
    return figure_to_base64(figure)


def interpretability_image(input_tensor: torch.Tensor) -> str:
    """Render the 28x28 input as grayscale for a simple interpretability view."""
    array = input_tensor.detach().cpu().numpy().squeeze()
    figure, axis = plt.subplots(figsize=(4, 4))
    axis.imshow(array, cmap='viridis')
    axis.axis('off')
    return figure_to_base64(figure)


def preprocess_image_bytes(image_bytes: bytes) -> torch.Tensor:
    """Convert raw image bytes to a normalized tensor for the model."""
    # The preprocessing steps from your original file are kept,
    # as they are specific to how the canvas drawing is handled.
    # A normalization step is added to match the training process.
    image = Image.open(BytesIO(image_bytes)).convert('L')
    img_array = np.array(image, dtype=np.float32)

    mask = img_array > 25 # Threshold to detect strokes
    if mask.any():
        ys, xs = np.where(mask)
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()
        cropped = img_array[y_min:y_max + 1, x_min:x_max + 1]
    else:
        cropped = img_array # Handle blank image

    # Resize to 20x20 box, preserving aspect ratio
    h, w = cropped.shape
    if max(h, w) > 0:
        scale = 20.0 / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = Image.fromarray(cropped).resize((new_w, new_h), Image.LANCZOS)
    else:
        resized = Image.new('L', (20, 20))

    # Pad to 28x28 and center
    canvas = Image.new('L', (28, 28))
    x_offset = (28 - resized.width) // 2
    y_offset = (28 - resized.height) // 2
    canvas.paste(resized, (x_offset, y_offset))

    # Apply same transformations as validation data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    tensor = transform(canvas).unsqueeze(0) # Add batch dimension
    return tensor


@torch.no_grad()
def mnist_prediction(img_tensor: torch.Tensor) -> Tuple[int, str, str]:
    """Run model inference and return (prediction, prob_plot_base64, interpretability_base64)."""
    img_tensor = img_tensor.to(DEVICE, dtype=torch.float)
    outputs = MODEL(x=img_tensor)
    
    # Convert logits to probabilities
    probabilities = (torch.softmax(outputs, dim=1)[0] * 100.0).cpu().numpy().tolist()
    
    prob_b64 = probability_plot_image(probabilities)
    interp_b64 = interpretability_image(img_tensor) # Use the normalized tensor for visualization
    _, output = torch.max(outputs.data, 1)
    prediction = int(output.cpu().numpy()[0])
    return prediction, prob_b64, interp_b64


@app.route('/process', methods=['POST'])
def process():
    data = request.get_data()
    if data.startswith(b'data:image/png;base64,'):
        img_bytes = base64.b64decode(data.split(b',')[1])
    else:
        return jsonify({'error': 'Invalid format'}), 400

    tensor = preprocess_image_bytes(img_bytes)
    pred, probencoded, interpretencoded = mnist_prediction(tensor)

    response = {
        'data': str(pred),
        'probencoded': probencoded,
        'interpretencoded': interpretencoded,
    }
    return jsonify(response)


@app.route('/', methods=['GET'])
def start():
    return render_template('default.html')


if __name__ == '__main__':
    # This block is now only for running the app directly with Flask's built-in server
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG_MODE)