# Research Paper Summarizer & QA Chatbot

A powerful, AI-driven web application to intelligently summarize complex academic research papers, chat with them, and convert summaries into speech.

## 1. 🧠 Core Model Explanation
**Model Architecture:** The backbone of this application is **`facebook/bart-large-cnn`**, which is based on the **BART (Bidirectional and Auto-Regressive Transformers)** architecture by Meta (Facebook AI).

**Why BART?** 
BART is specifically designed for sequence-to-sequence tasks like text summarization. It uses a standard Transformer architecture consisting of a Bidirectional Encoder (like BERT) and an Auto-Regressive Decoder (like GPT). 
- The encoder reads the massive, complex text from the research paper to capture deep contextual representations.
- The decoder takes those representations and generates a highly abstractive, condensed, human-readable summary.

We fine-tuned the model locally (and via Google Colab scripts) directly on structured research datasets, allowing it to specifically target and output distinct sections: *Methodology, Findings, Conclusion, and Limitations*.

## 2. 🔀 API Integration
*(Note: While the request mentioned FastAPI, the application is actually built and integrated utilizing a robust **Flask API** Server!)*

The ML model is served via `app.py` using **Flask API endpoints** to communicate directly with the Frontend UI seamlessly:
- **`POST /summarize`**: Handles the PDF uploads. When a user uploads a PDF, the Flask API extracts the raw text via `PyPDF2`, chunks it, and passes it into the BART model running in inference mode. It then returns the generated JSON summaries back to the dashboard UI.
- **REST Architecture**: The server uses Flask's `request` and `jsonify` functions to accept JSON parameters and files, ensuring smooth, non-blocking asynchronous calls via Cross-Origin Resource Sharing (CORS).

## 3. 🎙️ Text-To-Speech (TTS) Integration
**Engine Used**: **`pyttsx3`** (Python Text-to-Speech Version 3)

Unlike cloud-dependent APIs like Google Cloud TTS or ElevenLabs, we specifically integrated `pyttsx3` because it works entirely **offline**. 
**How it works:**
1. When you request a summary audio file, a call is made to the `POST /tts` Flask endpoint.
2. The `pyttsx3` library hooks directly into the native TTS engine of your operating system (SAPI5 for Windows).
3. It buffers the generated summary text and safely renders it down into high-quality `.wav` audio files.
4. The generated WAV file is dispatched directly back to the user’s browser for immediate playback.

## 4. 🤖 RAG Chatbot Integration
The chatbot relies on a **Retrieval-Augmented Generation (RAG)** pipeline. This allows the chatbot to answer highly specific questions regarding a paper you just uploaded *without* hallucinating.

**Integration Flow (`POST /chat`):**
1. **Context Extraction:** When you upload a paper via `/summarize`, the raw background text is saved temporarily in a `_chat_context` cache within the Flask server.
2. **Retrieval (Keyword Scoring):** When you ask a question in the chat panel, the script splits the entire massive research paper into individual sentences. It mathematically scores every sentence against the keywords within your question. 
3. **Augmentation:** It surgically extracts the **Top 5 most relevant sentences** representing the answer.
4. **Generation:** Instead of returning raw chunks, it combines these 5 sentences and constructs a special prompt: `Based on this context from a research paper, answer the question in simple language...`. This prompt is pushed to the BART model to generate a conversational, easy-to-read answer!
