"""
Configuration file for the Research Paper Summarizer.
All hyperparameters, paths, and settings are centralized here.
"""

import os

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR))  # parent folder with JSON files
INPUTS_FILE = os.path.join(DATA_DIR, "inputs.json")
OUTPUTS_FILE = os.path.join(DATA_DIR, "outputs - Copy.json")

MODEL_DIR = os.path.join(BASE_DIR, "models")
FINETUNED_MODEL_DIR = os.path.join(MODEL_DIR, "bart-research-summarizer")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
TTS_OUTPUT_DIR = os.path.join(BASE_DIR, "tts_output")

# Create directories
for d in [MODEL_DIR, FINETUNED_MODEL_DIR, LOGS_DIR, UPLOAD_DIR, TTS_OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

# ──────────────────────────────────────────────
# Model Configuration
# ──────────────────────────────────────────────
BASE_MODEL_NAME = "facebook/bart-large-cnn"

# Tokenizer settings
MAX_INPUT_LENGTH = 1024      # Max tokens for the input paper text
MAX_TARGET_LENGTH = 512      # Max tokens for the summary output

# ──────────────────────────────────────────────
# Training Hyperparameters
# ──────────────────────────────────────────────
TRAINING_CONFIG = {
    "num_train_epochs": 2,                # Lowered from 5 for practical local testing
    "per_device_train_batch_size": 1,     # Reduced for memory constraints
    "per_device_eval_batch_size": 1,      # Reduced for memory constraints
    "gradient_accumulation_steps": 8,     # Maintain effective batch size of 8
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_steps": 10,                   # Reduced since dataset is small
    "logging_steps": 5,                   # Log more often to see progress
    "eval_steps": 20,                     # Evaluate more often
    "save_steps": 20,
    "save_total_limit": 1,
    "fp16": False,                        # Disabled FP16 to prevent Windows CPU/CUDA mixed conflicts
    "eval_strategy": "steps",
    "load_best_model_at_end": True,
    "metric_for_best_model": "rouge2",
    "greater_is_better": True,
    "seed": 42,
    "dataloader_num_workers": 0,          # Must be 0 on Windows
    "report_to": "none",
}

# ──────────────────────────────────────────────
# Generation / Inference Configuration
# ──────────────────────────────────────────────
GENERATION_CONFIG = {
    "max_length": MAX_TARGET_LENGTH,
    "min_length": 100,
    "num_beams": 4,
    "length_penalty": 2.0,
    "early_stopping": True,
    "no_repeat_ngram_size": 3,
}

# ──────────────────────────────────────────────
# Summary Output Format
# ──────────────────────────────────────────────
SUMMARY_SECTIONS = [
    "summary",
    "methodology",
    "findings",
    "conclusion",
    "limitations",
]

# System prompt prepended to input text for better summarization
SUMMARIZE_PROMPT = (
    "Summarize the following research paper in simple language "
    "as if a teacher is explaining it to students. "
    "Cover the main idea, methodology, findings, conclusion, and limitations.\n\n"
)

# ──────────────────────────────────────────────
# TTS Configuration
# ──────────────────────────────────────────────
TTS_RATE = 160       # Words per minute
TTS_VOLUME = 1.0     # Volume (0.0 to 1.0)

# ──────────────────────────────────────────────
# Flask Server
# ──────────────────────────────────────────────
FLASK_HOST = os.environ.get("FLASK_HOST", "127.0.0.1")
FLASK_PORT = int(os.environ.get("PORT", 5000))
FLASK_DEBUG = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
