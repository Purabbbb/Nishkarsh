import os
import time
import logging
import json
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

from config import DATA_DIR, LOGS_DIR, TRAINING_CONFIG, MAX_TARGET_LENGTH, MAX_INPUT_LENGTH
from data_preprocessing import get_tokenized_datasets_for_model

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# List of models to compare
MODELS_TO_COMPARE = [
    "t5-small",
    "facebook/bart-base",
    "google/pegasus-xsum"
]

def compute_metrics(eval_pred, tokenizer):
    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")

    predictions, labels = eval_pred
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    # Handle empty predictions to prevent Rouge errors
    if not any(decoded_preds):
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    
    try:
        bertscore_result = bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
        result["bertscore_f1"] = float(np.mean(bertscore_result["f1"])) * 100
    except Exception as e:
        logger.warning(f"BERTScore failed: {e}")
        result["bertscore_f1"] = 0.0

    result = {key: float(value) * 100 for key, value in result.items() if isinstance(value, (int, float, np.number))}
    return {k: round(v, 4) for k, v in result.items()}


def finetune_and_evaluate(model_name):
    logger.info(f"--- Starting Evaluation for {model_name} ---")
    
    # Using a custom wrapper to get datasets for specific models since tokenizers differ
    from datasets import load_dataset
    inputs_file = os.path.join(DATA_DIR, "inputs.json")
    outputs_file = os.path.join(DATA_DIR, "outputs - Copy.json")
    
    with open(inputs_file, "r") as f:
        inputs = json.load(f)
    with open(outputs_file, "r") as f:
        outputs = json.load(f)
        
    # Create huggingface dataset format
    data_dict = {"document": [], "summary": []}
    for key in inputs:
        if key in outputs:
            data_dict["document"].append(inputs[key])
            data_dict["summary"].append(outputs[key])
            
    from datasets import Dataset
    hf_dataset = Dataset.from_dict(data_dict)
    hf_dataset = hf_dataset.train_test_split(test_size=0.2, seed=42)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Handle t5 prefix if needed
    prefix = "summarize: " if "t5" in model_name else ""
    
    def preprocess_function(examples):
        inputs_texts = [prefix + doc for doc in examples["document"]]
        model_inputs = tokenizer(inputs_texts, max_length=MAX_INPUT_LENGTH, truncation=True)
        labels = tokenizer(text_target=examples["summary"], max_length=MAX_TARGET_LENGTH, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
        
    tokenized_datasets = hf_dataset.map(preprocess_function, batched=True, remove_columns=["document", "summary"])
    train_dataset = tokenized_datasets["train"]
    val_dataset = tokenized_datasets["test"]
    
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    
    # Modify training args for faster comparison (e.g., fewer epochs, specific steps)
    compare_training_config = TRAINING_CONFIG.copy()
    compare_training_config["num_train_epochs"] = 1 # Quick run for comparison
    compare_training_config["max_steps"] = 20 # Only train for 20 steps to get a quick benchmark

    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join("models", model_name.replace("/", "-")),
        predict_with_generate=True,
        **compare_training_config
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer),
    )

    # Train and measure time
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    
    # Evaluate and measure time
    eval_start_time = time.time()
    eval_metrics = trainer.evaluate()
    eval_time = time.time() - eval_start_time
    
    results = {
        "model": model_name,
        "training_time_seconds": round(training_time, 2),
        "eval_time_seconds": round(eval_time, 2),
        "rouge1": eval_metrics.get("eval_rouge1", 0.0),
        "rouge2": eval_metrics.get("eval_rouge2", 0.0),
        "rougeL": eval_metrics.get("eval_rougeL", 0.0),
        "bertscore_f1": eval_metrics.get("eval_bertscore_f1", 0.0)
    }
    
    logger.info(f"Results for {model_name}: {results}")
    return results

if __name__ == "__main__":
    all_results = []
    for model_name in MODELS_TO_COMPARE:
        try:
            res = finetune_and_evaluate(model_name)
            all_results.append(res)
        except Exception as e:
            logger.error(f"Failed on {model_name}: {e}")
            
    # Save comparison to file
    with open("model_comparison_results.json", "w") as f:
        json.dump(all_results, f, indent=4)
        
    print("\n" + "="*50)
    print("COMPARISON RESULTS:")
    for r in all_results:
        print(f"Model: {r['model']}")
        print(f" - Train Time: {r['training_time_seconds']}s")
        print(f" - ROUGE-1: {r['rouge1']}")
        print(f" - ROUGE-2: {r['rouge2']}")
        print(f" - ROUGE-L: {r['rougeL']}")
        print(f" - BERTScore F1: {r['bertscore_f1']}")
        print("-" * 50)
