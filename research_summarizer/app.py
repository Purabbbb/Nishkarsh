"""
Flask API Server for the Research Paper Summarizer.
End-to-End implementation with Authentication and Database.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import transformers
transformers.utils.logging.disable_progress_bar()
import uuid
import logging
import json
from flask import Flask, request, jsonify, send_file, render_template, redirect, url_for, flash
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import login_user, login_required, logout_user, current_user

from config import FLASK_HOST, FLASK_PORT, FLASK_DEBUG, UPLOAD_DIR, TTS_OUTPUT_DIR
from inference import ResearchPaperSummarizer

from extensions import db, login_manager
from models import User, Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB max upload
app.secret_key = "super-secret-nishkarsh-key" # For sessions
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///nishkarsh.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize extensions
db.init_app(app)
login_manager.init_app(app)
login_manager.login_view = "login"

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Create tables if they don't exist
with app.app_context():
    db.create_all()

# Initialize summarizer once
logger.info("Initializing Research Paper Summarizer...")
summarizer = ResearchPaperSummarizer(use_finetuned=True)
logger.info("Summarizer ready!")

# Cache for chatbot context
_chat_context = {"text": None, "filename": None}


# ─── Page Routes ───────────────────────────────────

@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == "POST":
        email = request.form.get("email") # using email as username
        password = request.form.get("password")
        
        user = User.query.filter_by(username=email).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user, remember=True)
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid email or password", "error")
            
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
        
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        
        user = User.query.filter_by(username=email).first()
        if user:
            flash("Email already registered", "error")
        else:
            new_user = User(username=email, password_hash=generate_password_hash(password))
            db.session.add(new_user)
            db.session.commit()
            login_user(new_user, remember=True)
            return redirect(url_for('dashboard'))
            
    return render_template("register.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for('landing'))

@app.route("/dashboard")
@login_required
def dashboard():
    user_datasets = Dataset.query.filter_by(user_id=current_user.id).order_by(Dataset.created_at.desc()).all()
    return render_template("dashboard.html", datasets=user_datasets)

@app.route("/dataset/<int:dataset_id>")
@login_required
def view_dataset(dataset_id):
    dataset = Dataset.query.get_or_404(dataset_id)
    if dataset.user_id != current_user.id:
        return jsonify({"error": "Unauthorized"}), 403
    return jsonify({
        "id": dataset.id,
        "filename": dataset.filename,
        "summary": dataset.get_summary()
    })

# ─── API Routes ────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "BART-Large-CNN (Research Summarizer)"})

@app.route("/summarize", methods=["POST"])
def summarize():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Send a PDF with key 'file'."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are accepted."}), 400

    safe_name = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex[:8]}_{safe_name}"
    filepath = os.path.join(UPLOAD_DIR, unique_name)
    file.save(filepath)
    logger.info(f"Uploaded file saved: {filepath}")

    try:
        extracted_text = summarizer.extract_text_from_pdf(filepath)
        _chat_context["text"] = extracted_text
        _chat_context["filename"] = safe_name

        result = summarizer.summarize_pdf(filepath)
        
        # Save to DB if logged in
        if current_user.is_authenticated:
            new_dataset = Dataset(
                user_id=current_user.id,
                filename=safe_name,
                extracted_text=extracted_text
            )
            new_dataset.set_summary(result)
            db.session.add(new_dataset)
            db.session.commit()
            
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
        if os.path.exists(filepath):
            os.remove(filepath)

@app.route("/tts", methods=["POST"])
def text_to_speech():
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
        paper_text = _chat_context["text"]
        sentences = paper_text.replace("\n", " ").split(". ")

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
