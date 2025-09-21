import os
import requests
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Get the Backend URL from environment variables
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:3000")

@app.route("/")
def index():
    """Renders the main chat interface."""
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_files():
    """
    Forwards uploaded files (PDFs or images) to the FastAPI backend.
    """
    upload_mode = request.form.get("mode", "pdf")
    
    if "files" not in request.files:
        return jsonify({"error": "No files were uploaded"}), 400

    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files selected"}), 400

    # Prepare files for the backend request
    files_to_forward = []
    for file in files:
        files_to_forward.append(
            ("files", (file.filename, file.read(), file.mimetype))
        )

    try:
        if upload_mode == "image":
            upload_url = f"{BACKEND_URL}/upload/multiple/images"
        else:
            upload_url = f"{BACKEND_URL}/upload/pdfs"
            
        backend_response = requests.post(upload_url, files=files_to_forward, timeout=300)
        
        # Check if the response was successful
        backend_response.raise_for_status()

        return jsonify(backend_response.json()), backend_response.status_code

    except requests.exceptions.RequestException as e:
        error_message = f"Failed to connect to backend: {e}"
        # Try to get more specific error from response if available
        if e.response is not None:
            try:
                error_detail = e.response.json().get("detail", e.response.text)
                error_message = f"Backend Error: {error_detail}"
            except:
                error_message = f"Backend Error: {e.response.status_code} - {e.response.reason}"

        return jsonify({"error": error_message}), 502 # Bad Gateway

@app.route("/chat", methods=["POST"])
def chat():
    """
    Forwards a chat query to the FastAPI backend.
    """
    data = request.json
    chat_mode = data.get("mode")
    
    try:
        if chat_mode == "image":
            chat_url = f"{BACKEND_URL}/chat_image"
            payload = {
                "common_id": data.get("common_id"),
                "query": data.get("query")
            }
            # The image chat endpoint in the backend uses GET with JSON body, which is unusual.
            # A robust client should handle this. Using requests' `json` param with GET.
            backend_response = requests.get(chat_url, json=payload, timeout=60)
        else:
            collection_id = data.get("collection_id")
            chat_url = f"{BACKEND_URL}/chat/{collection_id}"
            payload = {
                "query": data.get("query"),
                "top_k": data.get("top_k", 3)
            }
            backend_response = requests.get(chat_url, json=payload, timeout=60)
        
        backend_response.raise_for_status()
        
        return jsonify(backend_response.json()), backend_response.status_code

    except requests.exceptions.RequestException as e:
        error_message = f"Failed to connect to backend for chat: {e}"
        if e.response is not None:
             try:
                error_detail = e.response.json().get("detail", e.response.text)
                error_message = f"Backend Chat Error: {error_detail}"
             except:
                error_message = f"Backend Chat Error: {e.response.status_code} - {e.response.reason}"
        return jsonify({"error": error_message}), 502

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8501, debug=False)
