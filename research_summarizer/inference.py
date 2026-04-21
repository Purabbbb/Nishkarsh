"""
Inference Pipeline
- Load fine-tuned BART model (or fallback to base model)
- Extract text from uploaded PDF
- Generate structured summary in simple language
- Text-to-Speech output
"""

import os
import logging
import torch
import pyttsx3
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from config import (
    BASE_MODEL_NAME, FINETUNED_MODEL_DIR,
    MAX_INPUT_LENGTH, GENERATION_CONFIG,
    SUMMARIZE_PROMPT, TTS_RATE, TTS_VOLUME, TTS_OUTPUT_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ResearchPaperSummarizer:
    """End-to-end research paper summarizer with TTS support."""

    def __init__(self, use_finetuned: bool = True):
        """
        Initialize the summarizer.
        Args:
            use_finetuned: If True, load the fine-tuned model. 
                          Falls back to base BART-CNN if fine-tuned model not found.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Decide which model to load
        model_path = FINETUNED_MODEL_DIR if use_finetuned else BASE_MODEL_NAME

        if use_finetuned and not os.path.exists(os.path.join(FINETUNED_MODEL_DIR, "config.json")):
            logger.warning(
                f"Fine-tuned model not found at {FINETUNED_MODEL_DIR}. "
                f"Falling back to base model: {BASE_MODEL_NAME}"
            )
            model_path = BASE_MODEL_NAME

        logger.info(f"Loading model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        self.model.eval()
        logger.info("Model loaded successfully!")

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract all text from a PDF file.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        reader = PdfReader(pdf_path)
        text_parts = []

        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text.strip())

        full_text = "\n".join(text_parts)
        logger.info(f"Extracted {len(full_text)} characters from {len(reader.pages)} pages")

        if not full_text.strip():
            raise ValueError("Could not extract any text from the PDF. It may be image-based.")

        return full_text

    def _chunk_text(self, text: str, chunk_size: int = 600) -> list:
        """
        Split long text into overlapping chunks to handle papers
        longer than the model's max input length (~600-700 words per 1024 tokens).
        """
        words = text.split()
        chunks = []
        overlap = chunk_size // 5

        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i : i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)

        return chunks

    def summarize_text(self, text: str, section_type: str = "general") -> str:
        """
        Generate a summary from input text for a specific section.
        """
        tokens = self.tokenizer.encode(text, truncation=False)

        if len(tokens) <= MAX_INPUT_LENGTH:
            return self._generate_summary(text, section_type)

        # For long texts: chunk and summarize (max 3 chunks to prevent long processing)
        chunks = self._chunk_text(text)
        chunk_summaries = []
        for i, chunk in enumerate(chunks[:3]):
            summary = self._generate_summary(chunk, section_type)
            chunk_summaries.append(summary)
            
        if len(chunk_summaries) == 1:
            return chunk_summaries[0]

        combined = " ".join(chunk_summaries)
        return self._generate_summary(combined, section_type)

    def _generate_summary(self, text: str, section_type: str) -> str:
        """
        Core generation function using BART with section-specific prompts.
        """
        prompts = {
            "summary": "Summarize this research paper in simple language for a student, focusing on the overall goal and big picture.",
            "methodology": "Explain the methodology or how this research was conducted in very simple terms.",
            "findings": "What were the key findings or results of this research? Explain them simply as if to a student.",
            "conclusion": "What is the final conclusion of this research paper? Explain it simply.",
            "limitations": "What were the limitations of this study? Explain why some things might not be perfect in simple words."
        }
        
        prompt = prompts.get(section_type, SUMMARIZE_PROMPT)
        input_text = prompt + "\n\n" + text

        inputs = self.tokenizer(
            input_text,
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **GENERATION_CONFIG,
            )

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary.strip()

    def summarize_pdf(self, pdf_path: str) -> dict:
        """
        Optimized pipeline: 
        1. Extract text
        2. Categorize sentences based on keywords (Extractive Summarization)
        3. Run abstractive summarization per section (Fast & distinct)
        """
        logger.info(f"Processing PDF (Fast Extractive + Abstractive): {pdf_path}")
        text = self.extract_text_from_pdf(pdf_path)
        
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
            
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)
            
        sentences = nltk.sent_tokenize(text)
        
        keywords = {
            "methodology": ["method", "approach", "procedure", "experiment", "design", "participant", "data collection", "measure", "setup", "evaluation", "framework"],
            "findings": ["result", "find", "found", "show", "demonstrate", "observe", "analysis", "significant", "performance", "outperform", "table", "figure"],
            "conclusion": ["conclude", "conclusion", "discuss", "summary", "future", "overall", "takeaway"],
            "limitations": ["limit", "restrict", "challenge", "shortcoming", "weakness", "fail", "unable", "future work", "drawback", "bias"]
        }
        
        categorized_text = {
            "methodology": [],
            "findings": [],
            "conclusion": [],
            "limitations": []
        }
        
        for s in sentences:
            s_lower = s.lower()
            for sec, kws in keywords.items():
                if any(kw in s_lower for kw in kws):
                    categorized_text[sec].append(s)
                    
        # Summary section -> use Intro + Abstract
        intro_words = text.split()[:1000]
        intro_text = " ".join(intro_words)
        
        # Conclusion fallback if keyword extraction is weak
        if len(" ".join(categorized_text["conclusion"]).split()) < 50:
            categorized_text["conclusion"] = [" ".join(text.split()[-1000:])]

        sections = {}
        
        logger.info("Generating General Summary...")
        sections["summary"] = self.summarize_text(intro_text, "summary")
        
        for sec in ["methodology", "findings", "conclusion", "limitations"]:
            joined = " ".join(categorized_text[sec])
            sec_words = joined.split()
            
            # Subsample if too long (max ~800 words)
            if len(sec_words) > 800:
                joined = " ".join(sec_words[:800])
            elif len(sec_words) < 20:
                # Fallback if specific section lacks text
                joined = intro_text
                
            logger.info(f"Generating {sec}...")
            sections[sec] = self.summarize_text(joined, sec)

        return {
            "pdf_path": pdf_path,
            "num_pages": len(PdfReader(pdf_path).pages),
            "extracted_chars": len(text),
            "sections": sections,
        }

    @staticmethod
    def text_to_speech(text: str, output_file: str = None, rate: int = TTS_RATE) -> str:
        """
        Convert summary text to speech.
        Args:
            text: The text to speak
            output_file: Optional path to save audio (.mp3/.wav)
            rate: Speech rate (words per minute)
        Returns:
            Path to saved audio file (if output_file provided), else None.
        """
        engine = pyttsx3.init()
        engine.setProperty("rate", rate)
        engine.setProperty("volume", TTS_VOLUME)

        # Try to select a good voice (prefer female for clarity)
        voices = engine.getProperty("voices")
        for voice in voices:
            if "zira" in voice.name.lower() or "female" in voice.name.lower():
                engine.setProperty("voice", voice.id)
                break

        if output_file:
            # Ensure .mp3 or .wav extension
            if not output_file.endswith((".mp3", ".wav")):
                output_file += ".wav"
            
            save_path = os.path.join(TTS_OUTPUT_DIR, os.path.basename(output_file))
            engine.save_to_file(text, save_path)
            engine.runAndWait()
            logger.info(f"Audio saved to: {save_path}")
            return save_path
        else:
            engine.say(text)
            engine.runAndWait()
            return None


if __name__ == "__main__":
    import sys

    summarizer = ResearchPaperSummarizer(use_finetuned=True)

    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        result = summarizer.summarize_pdf(pdf_path)
        print("\n" + "=" * 60)
        print("RESEARCH PAPER SUMMARY")
        print("=" * 60)
        print(f"Pages: {result['num_pages']}")
        print(f"Characters extracted: {result['extracted_chars']}")
        print("-" * 60)
        print("SUMMARY SECTION:")
        print(result["sections"]["summary"])
        print("-" * 60)
        print("FINDINGS SECTION:")
        print(result["sections"]["findings"])
        print("=" * 60)

        # TTS
        speak = input("\nWould you like to hear the summary? (y/n): ").strip().lower()
        if speak == "y":
            print("Speaking summary...")
            summarizer.text_to_speech(result["sections"]["summary"])
            print("Done!")
    else:
        print("Usage: python inference.py <path_to_pdf>")
        print("\nRunning quick test with sample text...")

        test_text = (
            "This paper presents a novel approach to text summarization using "
            "transformer-based models. The methodology involves fine-tuning BART "
            "on domain-specific datasets. Results show significant improvement in "
            "ROUGE scores compared to baseline models. The key limitation is the "
            "computational cost of training large models."
        )

        summary = summarizer.summarize_text(test_text)
        print(f"\nInput:\n{test_text}")
        print(f"\nSummary:\n{summary}")
