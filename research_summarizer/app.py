"""
Flask API Server for the Research Paper Summarizer.
Endpoints:
    GET  /              - Landing page
    GET  /login         - Login page
    GET  /dashboard     - Dashboard page
    POST /summarize     - Upload PDF, get structured summary
    POST /tts           - Generate TTS audio from text
    POST /chat          - RAG chatbot endpoint
    GET  /health        - Health check
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import transformers
transformers.utils.logging.disable_progress_bar()
import uuid
import logging
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

from config import FLASK_HOST, FLASK_PORT, FLASK_DEBUG, UPLOAD_DIR, TTS_OUTPUT_DIR
from inference import ResearchPaperSummarizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB max upload

# Initialize summarizer once
logger.info("Initializing Research Paper Summarizer...")
summarizer = ResearchPaperSummarizer(use_finetuned=True)
logger.info("Summarizer ready!")

# Cache for chatbot context (stores extracted text from most recent upload)
_chat_context = {"text": None, "filename": None}


# ─── Page Routes ───────────────────────────────────

@app.route("/")
def landing():
    """Serve the landing page."""
    return render_template("landing.html")


@app.route("/login")
def login():
    """Serve the login page."""
    return render_template("login.html")


@app.route("/dashboard")
def dashboard():
    """Serve the dashboard page."""
    return render_template("dashboard.html")


# ─── API Routes ────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "model": "BART-Large-CNN (Research Summarizer)"})


@app.route("/summarize", methods=["POST"])
def summarize():
    """
    Upload a PDF and get a structured summary.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Send a PDF with key 'file'."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are accepted."}), 400

    # Save uploaded file
    safe_name = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex[:8]}_{safe_name}"
    filepath = os.path.join(UPLOAD_DIR, unique_name)
    file.save(filepath)
    logger.info(f"Uploaded file saved: {filepath}")

    try:
        # Extract text for chatbot context
        extracted_text = summarizer.extract_text_from_pdf(filepath)
        _chat_context["text"] = extracted_text
        _chat_context["filename"] = safe_name

        result = summarizer.summarize_pdf(filepath)
        return jsonify({
            "success": True,
            "filename": safe_name,
            "num_pages": result["num_pages"],
            "extracted_chars": result["extracted_chars"],
            "sections": result["sections"],
        })
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route("/tts", methods=["POST"])
def text_to_speech():
    """
    Convert text to speech and return audio file.
    Body: { "text": "..." }
    """
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Send JSON with 'text' field."}), 400

    text = data["text"]
    if not text.strip():
        return jsonify({"error": "Text is empty."}), 400

    audio_filename = f"summary_{uuid.uuid4().hex[:8]}.wav"

    try:
        audio_path = summarizer.text_to_speech(text, output_file=audio_filename)
        return send_file(audio_path, mimetype="audio/wav", as_attachment=True, download_name=audio_filename)
    except Exception as e:
        logger.error(f"TTS failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/chat", methods=["POST"])
def chat():
    """
    RAG-style chatbot endpoint.
    Body: { "question": "..." }
    Uses the extracted text from the most recently uploaded paper.
    """
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Send JSON with 'question' field."}), 400

    question = data["question"].strip()
    if not question:
        return jsonify({"error": "Question is empty."}), 400

    if not _chat_context["text"]:
        return jsonify({
            "error": "No paper uploaded yet. Please upload a research paper first, then ask me questions about it."
        }), 400

    try:
        # Simple keyword-based retrieval from the paper text
        paper_text = _chat_context["text"]
        sentences = paper_text.replace("\n", " ").split(". ")

        # Score sentences by relevance to the question
        question_words = set(question.lower().split())
        scored = []
        for s in sentences:
            s_lower = s.lower()
            score = sum(1 for w in question_words if w in s_lower and len(w) > 2)
            if score > 0:
                scored.append((score, s.strip()))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_sentences = [s for _, s in scored[:5]]

        if top_sentences:
            context = ". ".join(top_sentences)
            # Use the model to generate a conversational answer
            prompt = f"Based on this context from a research paper, answer the question in simple language.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
            answer = summarizer.summarize_text(prompt, "general")
        else:
            answer = (
                f"I couldn't find specific information about that in the paper "
                f"'{_chat_context['filename']}'. Try asking about the methodology, "
                f"results, dataset, or conclusion."
            )

        return jsonify({"answer": answer})

    except Exception as e:
        logger.error(f"Chat failed: {e}")
        return jsonify({"error": f"Chat error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
