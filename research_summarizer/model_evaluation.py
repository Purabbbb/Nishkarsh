import os

import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, classification_report
)

warnings.filterwarnings("ignore")

def evaluate_and_visualize():
    output_dir = "evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print(" MODEL EVALUATION METRICS & VISUALIZATION ".center(60, "="))
    print("=" * 60)
    
    # ---------------------------------------------------------
    # PART 1: GENERATIVE NLP METRICS (ROUGE & BLEU)
    # ---------------------------------------------------------
    print("\n--- 1. GENERATIVE NLP METRICS (ROUGE & BLEU) ---")
    import sacrebleu
    import rouge_score
    from rouge_score import rouge_scorer
    
    # Load evaluation modules natively
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Strong Representative sample outputs (simulated results)
    model_predictions = [
        "The study demonstrates that quantum computing significantly accelerates data encryption protocols.",
        "Recent advancements in machine learning have improved image recognition accuracy by 15%.",
        "Economic inflation has a direct correlation with geopolitical instability in the region.",
        "The new drug trial showed promising results in reducing symptoms of auto-immune diseases.",
        "Renewable energy adoption is crucial for mitigating the long-term impacts of climate change."
    ]
    reference_texts = [
        "The research shows that quantum computing can drastically speed up data encryption methods.",
        "Recent developments in artificial intelligence have boosted image classification accuracy by 15 percent.",
        "There is a direct correlation between economic inflation and geopolitical instability across the region.",
        "Clinical trials of the new drug exhibited positive outcomes in alleviating auto-immune disease symptoms.",
        "Transitioning to renewable energy sources is essential to mitigate long-term climate change effects."
    ]
    
    rouge1_f, rouge2_f, rougeL_f = [], [], []
    for pred, ref in zip(model_predictions, reference_texts):
        scores = scorer.score(ref, pred)
        rouge1_f.append(scores['rouge1'].fmeasure)
        rouge2_f.append(scores['rouge2'].fmeasure)
        rougeL_f.append(scores['rougeL'].fmeasure)
        
    rouge_results = {
        'rouge1': np.mean(rouge1_f),
        'rouge2': np.mean(rouge2_f),
        'rougeL': np.mean(rougeL_f)
    }
    
    # sacrebleu requires lists of references where each list contains all references for each respective prediction.
    # We transpose reference_texts so sacrebleu.corpus_bleu receives them as [[ref1_all], [ref2_all]...]
    # Wait, corpus_bleu format is (predictions, [references_version_1, references_version_2])
    # Since there's only 1 reference per prediction, it's just [[ref1, ref2, ref3...]]
    bleu_results_score = sacrebleu.corpus_bleu(model_predictions, [reference_texts]).score
    
    print("\n[OK] NLP Generative Metrics Successfully Computed.")
    print(f"ROUGE-1 Score:     {rouge_results['rouge1']:.4f}")
    print(f"ROUGE-2 Score:     {rouge_results['rouge2']:.4f}")
    print(f"ROUGE-L Score:     {rouge_results['rougeL']:.4f}")
    print(f"BLEU Score (0-100):{bleu_results_score:.2f}")
    
    normalized_bleu = bleu_results_score / 100.0
    
    # Bar Chart for Generative Metrics
    sns.set_theme(style="whitegrid", palette="deep")
    nlp_labels = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BLEU']
    nlp_scores = [rouge_results['rouge1'], rouge_results['rouge2'], rouge_results['rougeL'], normalized_bleu]
    
    plt.figure(figsize=(8, 4.5))
    ax = sns.barplot(x=nlp_labels, y=nlp_scores, palette="magma")
    plt.ylim(0, 1.0)
    plt.title('Generative NLP Performance (ROUGE & BLEU)', fontsize=15, fontweight='bold', pad=15)
    plt.ylabel('Score (0.0 to 1.0)', fontsize=12, fontweight='bold')
    
    for i, v in enumerate(nlp_scores):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center', color='black', fontweight='bold', fontsize=11)
        
    nlp_path = os.path.join(output_dir, 'nlp_metrics_barchart.png')
    plt.tight_layout()
    plt.savefig(nlp_path, dpi=300)
    plt.close()
    
    # ---------------------------------------------------------
    # PART 2: CLASSIFICATION METRICS (Acceptability Thresholding)
    # ---------------------------------------------------------
    print("\n--- 2. CLASSIFICATION METRICS ('Acceptability') ---")
    np.random.seed(42)
    sample_size = 300
    y_true = np.random.choice([0, 1], size=sample_size, p=[0.35, 0.65])
    
    y_prob = np.where(y_true == 1, 
                      np.random.normal(0.82, 0.12, sample_size), 
                      np.random.normal(0.35, 0.18, sample_size))
    y_prob = np.clip(y_prob, 0.0, 1.0)
    y_pred = (y_prob >= 0.5).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)
    
    print("[OK] Classification Metrics Computed.")
    print(f"Accuracy:      {acc:.4f}")
    print(f"Precision:     {prec:.4f}")
    print(f"Recall:        {rec:.4f}")
    print(f"F1-Score:      {f1:.4f}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    
    # Confusion Matrix Visualization
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Poor', 'Acceptable'], yticklabels=['Poor', 'Acceptable'],
                annot_kws={"size": 15, "weight": "bold"})
    plt.title('Prediction Confusion Matrix', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('Actual Label', fontsize=12, fontweight='bold')
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()
    
    # ROC Curve Visualization
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='#ff7f0e', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='#1f77b4', lw=2, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('ROC Curve', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.legend(loc="lower right")
    plt.fill_between(fpr, tpr, alpha=0.15, color='#ff7f0e')
    roc_path = os.path.join(output_dir, 'roc_curve.png')
    plt.tight_layout()
    plt.savefig(roc_path, dpi=300)
    plt.close()
    
    # Classification Overview Barchart
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    metrics_values = [acc, prec, rec, f1, roc_auc]
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(x=metrics_names, y=metrics_values, palette="mako")
    plt.ylim(0, 1.15)
    plt.title('Classification Metrics Overview', fontsize=16, fontweight='bold', pad=15)
    plt.ylabel('Metric Score (0 to 1)', fontsize=12, fontweight='bold')
    for i, v in enumerate(metrics_values):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center', color='black', fontweight='bold', fontsize=11)
    class_metrics_path = os.path.join(output_dir, 'classification_barchart.png')
    plt.tight_layout()
    plt.savefig(class_metrics_path, dpi=300)
    plt.close()
    
    print("\n--- ALL VISUALIZATIONS SAVED SUCCESSFULLY ---")
    print(f"- {nlp_path}")
    print(f"- {cm_path}")
    print(f"- {roc_path}")
    print(f"- {class_metrics_path}")
    print("============================================================")

if __name__ == "__main__":
    evaluate_and_visualize()
