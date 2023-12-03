"""
Flask Image Classification API

This Flask application serves as an API for classifying images using a pre-trained model. It provides a single endpoint
for image classification.

Routes:
- /classify_image (POST): Classify an image and return the result as JSON.

Functions:
- classify_image(): Endpoint for classifying an image and returning the result as JSON.

Globals:
- app: Flask application instance.

Usage:
1. Start the Flask server using 'python app.py'.
2. Send a POST request to '/classify_image' with a base64-encoded image data in the 'image_data' field of the request form.

Requirements:
- Flask
- util (Custom module for image classification)
"""

from flask import Flask, request, jsonify, render_template
import util
import os

if os.getcwd().endswith('server'):
    os.chdir('../')

app = Flask(__name__)

@app.route("/classify_image", methods=["POST"])
def classify_image():
    """
    Classify an image and return the result as JSON.
    
    Args:
        None (accesses image_data from request.form)

    Returns:
        JSON response with the classification result.
    """
    image_data = request.form["image_data"]
    response = jsonify(util.classify_image(image_data))
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

if __name__ == "__main__":
    print("Starting Python Flask Server For Sports Celebrity Image Classification")
    util.load_saved_artifacts()
    app.run(port=5000, debug=True)
