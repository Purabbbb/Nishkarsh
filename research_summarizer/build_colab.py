import os
import json

def build_notebook():
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# \ud83d\ude80 Research Paper Summarizer - Colab Fine-Tuning Pipeline\n",
                "This notebook contains the complete pipeline to train the BART model on a high-end cloud GPU for free using Google Colab.\n",
                "\n",
                "### \u26a0\ufe0f Prerequisites:\n",
                "1. On the left sidebar, click the **Folder** icon.\n",
                "2. Upload your `inputs.json` and `outputs - Copy.json` files.\n",
                "3. At the top right, click **Connect**. Then go to **Runtime > Change runtime type** and ensure **T4 GPU** is selected."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 1. Install Dependencies"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!pip install -q transformers evaluate rouge_score datasets PyPDF2 pyttsx3 accelerate\n",
                "import nltk\n",
                "nltk.download('punkt', quiet=True)\n",
                "nltk.download('punkt_tab', quiet=True)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 2. Load and Preprocess Data"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import json\n",
                "import datasets\n",
                "from transformers import AutoTokenizer\n",
                "\n",
                "MODEL_NAME = \"facebook/bart-large-cnn\"\n",
                "tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)\n",
                "\n",
                "def load_and_match_data(inputs_file, outputs_file):\n",
                "    with open(inputs_file, 'r', encoding='utf-8') as f:\n",
                "        inputs_data = json.load(f)\n",
                "    with open(outputs_file, 'r', encoding='utf-8') as f:\n",
                "        outputs_data = json.load(f)\n",
                "        \n",
                "    inputs_dict = {item['title']: item['full_text'] for item in inputs_data}\n",
                "    matches = []\n",
                "\n",
                "    for out_item in outputs_data:\n",
                "        title = out_item.get(\"title\")\n",
                "        if title in inputs_dict:\n",
                "            # Combine sections into a targeted output\n",
                "            combined_summary = (f\"SUMMARY: {out_item.get('summary', '')}\\n\"\n",
                "                                f\"METHODOLOGY: {out_item.get('methodology', '')}\\n\"\n",
                "                                f\"FINDINGS: {out_item.get('findings', '')}\\n\"\n",
                "                                f\"CONCLUSION: {out_item.get('conclusion', '')}\\n\"\n",
                "                                f\"LIMITATIONS: {out_item.get('limitations', '')}\")\n",
                "            matches.append({\n",
                "                \"input_text\": f\"Summarize the following research paper:\\n\\n{inputs_dict[title]}\",\n",
                "                \"target_text\": combined_summary\n",
                "            })\n",
                "    print(f\"Matched {len(matches)} pairs!\")\n",
                "    return datasets.Dataset.from_list(matches)\n",
                "\n",
                "df = load_and_match_data(\"inputs.json\", \"outputs - Copy.json\")\n",
                "df = df.train_test_split(test_size=0.15, seed=42)\n",
                "train_dataset = df['train']\n",
                "val_dataset = df['test']\n",
                "\n",
                "def tokenize_fn(batch):\n",
                "    model_inputs = tokenizer(batch[\"input_text\"], max_length=1024, truncation=True, padding=\"max_length\")\n",
                "    labels = tokenizer(batch[\"target_text\"], max_length=512, truncation=True, padding=\"max_length\")\n",
                "    # Ignore padding for loss\n",
                "    model_inputs[\"labels\"] = [\n",
                "        [(l if l != tokenizer.pad_token_id else -100) for l in label]\n",
                "        for label in labels[\"input_ids\"]\n",
                "    ]\n",
                "    return model_inputs\n",
                "\n",
                "train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=[\"input_text\", \"target_text\"])\n",
                "val_dataset = val_dataset.map(tokenize_fn, batched=True, remove_columns=[\"input_text\", \"target_text\"])\n",
                "print(\"Tokenization Complete!\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 3. Define Evaluation Metrics (ROUGE)"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import evaluate\n",
                "import numpy as np\n",
                "\n",
                "rouge = evaluate.load(\"rouge\")\n",
                "\n",
                "def compute_metrics(eval_pred):\n",
                "    predictions, labels = eval_pred\n",
                "    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)\n",
                "    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)\n",
                "    \n",
                "    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)\n",
                "    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)\n",
                "\n",
                "    decoded_preds = [\"\\n\".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]\n",
                "    decoded_labels = [\"\\n\".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]\n",
                "\n",
                "    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)\n",
                "    result = {key: float(value) * 100 for key, value in result.items() if isinstance(value, (int, float, np.number))}\n",
                "    return {k: round(v, 4) for k, v in result.items()}"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 4. Fine-Tune the Model\n", "*This uses the GPU to train much faster than a local PC.*"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq\n",
                "import torch\n",
                "\n",
                "model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)\n",
                "\n",
                "training_args = Seq2SeqTrainingArguments(\n",
                "    output_dir=\"./bart-research-finetuned\",\n",
                "    predict_with_generate=True,\n",
                "    evaluation_strategy=\"epoch\",\n",
                "    save_strategy=\"epoch\",\n",
                "    learning_rate=3e-5,\n",
                "    per_device_train_batch_size=4,  # Larger batch size possible on Colab T4 GPU\n",
                "    per_device_eval_batch_size=4,\n",
                "    gradient_accumulation_steps=2,  # Effective batch = 8\n",
                "    weight_decay=0.01,\n",
                "    save_total_limit=2,\n",
                "    num_train_epochs=5,             # 5 Epochs for better learning\n",
                "    fp16=True,                      # Ultra-fast mixed precision on GPU\n",
                "    load_best_model_at_end=True,\n",
                "    metric_for_best_model=\"rouge2\"\n",
                ")\n",
                "\n",
                "data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)\n",
                "\n",
                "trainer = Seq2SeqTrainer(\n",
                "    model=model,\n",
                "    args=training_args,\n",
                "    train_dataset=train_dataset,\n",
                "    eval_dataset=val_dataset,\n",
                "    processing_class=tokenizer,\n",
                "    data_collator=data_collator,\n",
                "    compute_metrics=compute_metrics,\n",
                ")\n",
                "\n",
                "trainer.train()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 5. Evaluate and Export Model\n", "Save the model and run evaluation to see your new ROUGE scores."]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\"Final Evaluation Scores:\")\n",
                "print(trainer.evaluate())\n",
                "\n",
                "# Save model locally in Colab\n",
                "trainer.save_model(\"./final_model\")\n",
                "tokenizer.save_pretrained(\"./final_model\")\n",
                "\n",
                "# Zip it up to download\n",
                "!zip -r final_model.zip ./final_model\n",
                "print(\"\\n\\u2705 Done! You can now download final_model.zip from the filesystem on the left.\")"
            ]
        }
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"}
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }

    path = r"c:\Users\bhati\OneDrive\Desktop\pp\research_summarizer\Colab_Training_Pipeline.ipynb"
    with open(path, "w", encoding='utf-8') as f:
        json.dump(notebook, f, indent=2)

if __name__ == "__main__":
    build_notebook()
