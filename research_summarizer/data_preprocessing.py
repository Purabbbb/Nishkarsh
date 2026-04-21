"""
Data Preprocessing Pipeline
- Loads inputs.json and outputs - Copy.json
- Matches them by ID
- Creates train/validation splits
- Prepares HuggingFace Dataset for BART fine-tuning
"""

import json
import logging
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer

from config import (
    INPUTS_FILE, OUTPUTS_FILE, BASE_MODEL_NAME,
    MAX_INPUT_LENGTH, MAX_TARGET_LENGTH, SUMMARIZE_PROMPT,
    SUMMARY_SECTIONS
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_json(filepath: str) -> list:
    """Load and return JSON data from file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} entries from {filepath}")
    return data


def build_target_summary(output_entry: dict) -> str:
    """
    Combine all summary sections from the output entry into a single
    structured target string for training.
    """
    parts = []
    section_labels = {
        "summary": "Overview",
        "methodology": "Methodology",
        "findings": "Key Findings",
        "conclusion": "Conclusion",
        "limitations": "Limitations",
    }

    for section in SUMMARY_SECTIONS:
        if section in output_entry and output_entry[section].strip():
            label = section_labels.get(section, section.title())
            parts.append(f"**{label}**: {output_entry[section].strip()}")

    return "\n\n".join(parts)


def match_inputs_outputs(inputs: list, outputs: list) -> list:
    """
    Match input papers with their corresponding output summaries by ID.
    Returns a list of paired examples.
    """
    output_map = {entry["id"]: entry for entry in outputs}
    paired = []

    for inp in inputs:
        inp_id = inp.get("id")
        if inp_id in output_map:
            out = output_map[inp_id]
            source_text = inp.get("text", "").strip()
            target_summary = build_target_summary(out)

            if source_text and target_summary:
                paired.append({
                    "id": inp_id,
                    "title": inp.get("title", ""),
                    "source": source_text,
                    "target": target_summary,
                })

    logger.info(f"Matched {len(paired)} input-output pairs out of {len(inputs)} inputs and {len(outputs)} outputs")
    return paired


def prepare_datasets(test_size: float = 0.15, seed: int = 42) -> tuple:
    """
    Full pipeline: load → match → split → return HuggingFace Datasets.
    """
    inputs = load_json(INPUTS_FILE)
    outputs = load_json(OUTPUTS_FILE)
    paired = match_inputs_outputs(inputs, outputs)

    if len(paired) == 0:
        raise ValueError("No matched pairs found! Check your input/output JSON files.")

    train_data, val_data = train_test_split(
        paired, test_size=test_size, random_state=seed
    )

    logger.info(f"Train set: {len(train_data)} | Validation set: {len(val_data)}")

    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    return train_dataset, val_dataset


def tokenize_function(examples, tokenizer):
    """
    Tokenize source (input paper) and target (summary) for BART.
    Prepends a summarization prompt to the input.
    """
    # Prepend the prompt to each source text
    inputs = [SUMMARIZE_PROMPT + text for text in examples["source"]]

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length",
    )

    # Tokenize targets
    labels = tokenizer(
        text_target=examples["target"],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding="max_length",
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def get_tokenized_datasets():
    """
    Returns tokenized train and validation datasets ready for training.
    """
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    train_dataset, val_dataset = prepare_datasets()

    logger.info("Tokenizing training set...")
    train_tokenized = train_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train",
    )

    logger.info("Tokenizing validation set...")
    val_tokenized = val_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Tokenizing val",
    )

    return train_tokenized, val_tokenized, tokenizer


if __name__ == "__main__":
    # Quick test
    train_ds, val_ds = prepare_datasets()
    print(f"\n{'='*60}")
    print(f"Train: {len(train_ds)} examples")
    print(f"Val:   {len(val_ds)} examples")
    print(f"{'='*60}")
    print(f"\nSample source (first 300 chars):\n{train_ds[0]['source'][:300]}...")
    print(f"\nSample target (first 300 chars):\n{train_ds[0]['target'][:300]}...")
