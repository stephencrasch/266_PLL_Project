"""
Evaluate a trained ModernBERT model on the holdout set.
This script loads the holdout data and evaluates model performance.
"""

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from datasets import Dataset
import numpy as np
from tqdm import tqdm
import json
import sys


class HoldoutEvaluator:
    """Evaluate trained models on holdout set."""
    
    def __init__(self, model_path: str, holdout_csv: str = 'reviews_holdout.csv'):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to trained model checkpoint (e.g., './modernbert_r0')
            holdout_csv: Path to holdout CSV file
        """
        self.model_path = model_path
        self.holdout_csv = holdout_csv
        
        # Setup device
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        print(f"\nLoading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            trust_remote_code=True
        ).to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully")
        print(f"Number of labels: {self.model.config.num_labels}")
        print(f"Labels: {list(self.model.config.id2label.values())}")
        
        # Load holdout data
        print(f"\nLoading holdout data from {holdout_csv}...")
        self.holdout_df = pd.read_csv(holdout_csv)
        print(f"Loaded {len(self.holdout_df):,} holdout reviews")
        
        if 'label' in self.holdout_df.columns:
            print(f"\nHoldout set label distribution:")
            print(self.holdout_df['label'].value_counts().sort_index())
        else:
            raise ValueError("Holdout CSV must have a 'label' column!")
    
    def predict(self, texts: list, batch_size: int = 16) -> tuple:
        """
        Get predictions for a list of texts.
        
        Args:
            texts: List of review texts
            batch_size: Batch size for inference
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        all_preds = []
        all_probs = []
        
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                preds = logits.argmax(dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_preds), np.array(all_probs)
    
    def evaluate(self, batch_size: int = 16, save_results: bool = True) -> dict:
        """
        Evaluate model on holdout set.
        
        Args:
            batch_size: Batch size for inference
            save_results: Whether to save results to JSON
            
        Returns:
            Dictionary with evaluation metrics
        """
        print("\n" + "=" * 80)
        print("EVALUATING ON HOLDOUT SET")
        print("=" * 80)
        
        # Get texts and true labels
        texts = self.holdout_df['text'].tolist()
        true_labels_str = self.holdout_df['label'].tolist()
        
        # Map string labels to IDs
        label2id = self.model.config.label2id
        try:
            true_labels = [label2id[lbl] for lbl in true_labels_str]
        except KeyError as e:
            print(f"\n⚠️  ERROR: Label {e} not found in model's label mapping!")
            print(f"Model knows these labels: {list(label2id.keys())}")
            print(f"Holdout set has these labels: {set(true_labels_str)}")
            sys.exit(1)
        
        # Get predictions
        pred_labels, pred_probs = self.predict(texts, batch_size=batch_size)
        
        # Convert predictions back to label names
        id2label = self.model.config.id2label
        pred_labels_str = [id2label[pred] for pred in pred_labels]
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, pred_labels)
        f1_macro = f1_score(true_labels, pred_labels, average='macro')
        f1_weighted = f1_score(true_labels, pred_labels, average='weighted')
        precision_macro = precision_score(true_labels, pred_labels, average='macro')
        recall_macro = recall_score(true_labels, pred_labels, average='macro')
        
        # Print overall metrics
        print("\n" + "=" * 80)
        print("OVERALL METRICS")
        print("=" * 80)
        print(f"Accuracy:           {accuracy:.4f}")
        print(f"F1 (macro):         {f1_macro:.4f}")
        print(f"F1 (weighted):      {f1_weighted:.4f}")
        print(f"Precision (macro):  {precision_macro:.4f}")
        print(f"Recall (macro):     {recall_macro:.4f}")
        
        # Per-class metrics
        print("\n" + "=" * 80)
        print("PER-CLASS METRICS")
        print("=" * 80)
        print(classification_report(
            true_labels_str,
            pred_labels_str,
            digits=4
        ))
        
        # Confusion matrix
        print("\n" + "=" * 80)
        print("CONFUSION MATRIX")
        print("=" * 80)
        cm = confusion_matrix(true_labels, pred_labels)
        labels = [id2label[i] for i in range(len(id2label))]
        
        # Print confusion matrix with labels
        print("\nTrue \\ Predicted:", end="")
        for label in labels:
            print(f"{label:>15}", end="")
        print()
        
        for i, label in enumerate(labels):
            print(f"{label:>15}:", end="")
            for j in range(len(labels)):
                print(f"{cm[i][j]:>15}", end="")
            print()
        
        # Prepare results dictionary
        results = {
            "model_path": self.model_path,
            "holdout_csv": self.holdout_csv,
            "total_samples": len(self.holdout_df),
            "metrics": {
                "accuracy": float(accuracy),
                "f1_macro": float(f1_macro),
                "f1_weighted": float(f1_weighted),
                "precision_macro": float(precision_macro),
                "recall_macro": float(recall_macro)
            },
            "per_class_metrics": {},
            "confusion_matrix": cm.tolist(),
            "label_names": labels
        }
        
        # Add per-class metrics
        for label_name in labels:
            label_id = label2id[label_name]
            mask_true = np.array(true_labels) == label_id
            mask_pred = pred_labels == label_id
            
            tp = np.sum(mask_true & mask_pred)
            fp = np.sum(~mask_true & mask_pred)
            fn = np.sum(mask_true & ~mask_pred)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            support = int(np.sum(mask_true))
            
            results["per_class_metrics"][label_name] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": support
            }
        
        # Save results
        if save_results:
            # Extract round number from model path
            import re
            round_match = re.search(r'_r(\d+)', self.model_path)
            round_num = round_match.group(1) if round_match else '0'
            
            output_file = f"holdout_eval_r{round_num}.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n✅ Results saved to {output_file}")
        
        return results
    
    def predict_samples(self, n_samples: int = 10, random_state: int = 42):
        """
        Show predictions for random samples.
        
        Args:
            n_samples: Number of random samples to show
            random_state: Random seed
        """
        print("\n" + "=" * 80)
        print(f"SAMPLE PREDICTIONS (n={n_samples})")
        print("=" * 80)
        
        # Random sample
        sample_df = self.holdout_df.sample(n=min(n_samples, len(self.holdout_df)), random_state=random_state)
        
        texts = sample_df['text'].tolist()
        true_labels = sample_df['label'].tolist()
        
        # Get predictions
        pred_labels, pred_probs = self.predict(texts, batch_size=len(texts))
        
        # Show results
        id2label = self.model.config.id2label
        for i, (text, true_label) in enumerate(zip(texts, true_labels)):
            pred_label = id2label[pred_labels[i]]
            confidence = pred_probs[i][pred_labels[i]]
            
            print(f"\n--- Sample {i+1} ---")
            print(f"Text: {text[:150]}...")
            print(f"True Label:      {true_label}")
            print(f"Predicted Label: {pred_label}")
            print(f"Confidence:      {confidence:.4f}")
            print(f"Correct: {'✅' if pred_label == true_label else '❌'}")


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained model on test set')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model (e.g., ./modernbert_r0)')
    parser.add_argument('--test_csv', type=str, default='reviews_test.csv',
                       help='Path to test CSV file (default: reviews_test.csv)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for inference')
    parser.add_argument('--show_samples', type=int, default=0,
                       help='Number of random samples to show (0 = none)')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = HoldoutEvaluator(
        model_path=args.model_path,
        holdout_csv=args.test_csv
    )
    
    # Run evaluation
    results = evaluator.evaluate(batch_size=args.batch_size, save_results=True)
    
    # Show sample predictions if requested
    if args.show_samples > 0:
        evaluator.predict_samples(n_samples=args.show_samples)
    
    print("\n" + "=" * 80)
    print("✅ EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
