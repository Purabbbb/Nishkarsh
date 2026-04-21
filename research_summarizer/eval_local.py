import logging
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from data_preprocessing import get_tokenized_datasets
from config import BASE_MODEL_NAME, MAX_TARGET_LENGTH
from train import compute_metrics
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_eval():
    logger.info("loading data and model for custom evaluation...")
    train_dataset, val_dataset, tokenizer = get_tokenized_datasets()
    
    # We use a 3 sample subset here to get instant scores and skip wait time
    subset_size = min(3, len(val_dataset))
    val_dataset = val_dataset.select(range(subset_size))
    
    logger.info(f"Evaluating on {subset_size} local validation samples...")

    # Load base model since training has not been run
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_NAME)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir="./eval_tmp",
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,
        per_device_eval_batch_size=2,
        report_to="none",
        log_level="error"
    )
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer),
    )
    
    logger.info("Running custom generative ROUGE evaluation. This may take 2-4 minutes...")
    metrics = trainer.evaluate()
    
    print("\n" + "="*50)
    print("CUSTOM LOCAL EVALUATION SCORES (ROUGE)")
    print("="*50)
    print(f"ROUGE-1 (Word accuracy):   {metrics.get('eval_rouge1', 'N/A')}")
    print(f"ROUGE-2 (Phrase accuracy): {metrics.get('eval_rouge2', 'N/A')}")
    print(f"ROUGE-L (Sentence flow):   {metrics.get('eval_rougeL', 'N/A')}")
    print(f"ROUGE-Lsum (Summary flow): {metrics.get('eval_rougeLsum', 'N/A')}")
    print("==================================================")

if __name__ == "__main__":
    run_eval()
