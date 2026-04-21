"""
Fine-Tuning Script for facebook/bart-large-cnn
- Loads matched & tokenized dataset
- Configures Seq2SeqTrainer with ROUGE evaluation
- Saves the best model checkpoint
"""

import logging
import numpy as np
import nltk
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

import evaluate
from config import (
    BASE_MODEL_NAME, FINETUNED_MODEL_DIR, LOGS_DIR,
    TRAINING_CONFIG, MAX_TARGET_LENGTH,
)
from data_preprocessing import get_tokenized_datasets

# Download punkt tokenizer for sentence splitting in ROUGE
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_metrics(eval_pred, tokenizer):
    """Compute ROUGE metrics for evaluation."""
    rouge = evaluate.load("rouge")


    predictions, labels = eval_pred
    
    # Replace -100 in predictions and labels (padding) with pad_token_id
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Add newlines for ROUGE sentence-level scoring
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True,
    )

    # Extract scores (evaluate returns floats)
    result = {key: float(value) * 100 for key, value in result.items() if isinstance(value, (int, float, np.number))}

    # Add average generation length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in eval_pred[0]]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


def train():
    """Main training loop."""
    logger.info("=" * 60)
    logger.info("RESEARCH PAPER SUMMARIZER - Fine-Tuning BART-Large-CNN")
    logger.info("=" * 60)

    # 1. Load tokenized data
    logger.info("Loading and tokenizing datasets...")
    train_dataset, val_dataset, tokenizer = get_tokenized_datasets()
    logger.info(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # 2. Load model
    logger.info(f"Loading model: {BASE_MODEL_NAME}")
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_NAME)
    logger.info(f"Model parameters: {model.num_parameters():,}")

    # 3. Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    # 4. Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=FINETUNED_MODEL_DIR,
        logging_dir=LOGS_DIR,
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,
        **TRAINING_CONFIG,
    )

    # 5. Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer),
    )

    # 6. Train!
    logger.info("Starting training...")
    train_result = trainer.train()

    # 7. Save the best model
    logger.info(f"Saving fine-tuned model to: {FINETUNED_MODEL_DIR}")
    trainer.save_model(FINETUNED_MODEL_DIR)
    tokenizer.save_pretrained(FINETUNED_MODEL_DIR)

    # 8. Log final metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # 9. Evaluate
    logger.info("Running final evaluation...")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"Model saved to: {FINETUNED_MODEL_DIR}")
    logger.info("=" * 60)

    return trainer


if __name__ == "__main__":
    train()
