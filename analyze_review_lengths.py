"""
Analyze review text lengths and token distributions.
This helps determine if we need a longer-context embedding model or truncation strategy.
"""

import pandas as pd
import numpy as np
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from collections import Counter

def analyze_review_lengths(csv_path='reviews_for_embeddings.csv'):
    """
    Analyze review text lengths in characters and tokens.
    
    Args:
        csv_path: Path to the reviews CSV file
    """
    print("=" * 80)
    print("REVIEW LENGTH ANALYSIS")
    print("=" * 80)
    
    # Load data
    print(f"\nLoading reviews from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} reviews")
    
    # Character-level analysis
    print("\n" + "=" * 80)
    print("CHARACTER LENGTH ANALYSIS")
    print("=" * 80)
    
    df['char_length'] = df['text'].str.len()
    
    print(f"\nCharacter Statistics:")
    print(f"  Mean:   {df['char_length'].mean():.0f} chars")
    print(f"  Median: {df['char_length'].median():.0f} chars")
    print(f"  Std:    {df['char_length'].std():.0f} chars")
    print(f"  Min:    {df['char_length'].min():.0f} chars")
    print(f"  Max:    {df['char_length'].max():.0f} chars")
    
    # Percentiles
    print(f"\nPercentiles:")
    for p in [25, 50, 75, 90, 95, 99]:
        val = df['char_length'].quantile(p/100)
        print(f"  {p}th: {val:.0f} chars")
    
    # Distribution bins
    print(f"\nCharacter Length Distribution:")
    bins = [0, 250, 500, 750, 1000, 1500, 2000, 3000, 10000]
    labels = ['0-250', '250-500', '500-750', '750-1000', '1000-1500', '1500-2000', '2000-3000', '3000+']
    df['char_bin'] = pd.cut(df['char_length'], bins=bins, labels=labels)
    char_dist = df['char_bin'].value_counts().sort_index()
    for label, count in char_dist.items():
        pct = (count / len(df)) * 100
        print(f"  {label:12s}: {count:6,} ({pct:5.1f}%)")
    
    # Token-level analysis with multiple tokenizers
    print("\n" + "=" * 80)
    print("TOKEN LENGTH ANALYSIS")
    print("=" * 80)
    
    models_to_test = [
        ('all-MiniLM-L6-v2', 'sentence-transformers/all-MiniLM-L6-v2', 256),
        ('all-mpnet-base-v2', 'sentence-transformers/all-mpnet-base-v2', 384),
        ('bge-base-en-v1.5', 'BAAI/bge-base-en-v1.5', 512),
        ('distilbert', 'distilbert-base-uncased', 512),
    ]
    
    results = {}
    
    for model_name, model_path, max_tokens in models_to_test:
        print(f"\n{'-' * 80}")
        print(f"Model: {model_name} (max_seq_length: {max_tokens})")
        print(f"{'-' * 80}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Sample for speed (tokenization can be slow)
            sample_size = min(5000, len(df))
            sample_df = df.sample(n=sample_size, random_state=42)
            
            print(f"Tokenizing {sample_size:,} sample reviews...")
            
            # Tokenize without truncation to get true lengths
            token_lengths = []
            for text in sample_df['text']:
                tokens = tokenizer.encode(str(text), add_special_tokens=True, truncation=False)
                token_lengths.append(len(tokens))
            
            token_lengths = np.array(token_lengths)
            
            # Statistics
            print(f"\nToken Statistics:")
            print(f"  Mean:   {token_lengths.mean():.0f} tokens")
            print(f"  Median: {np.median(token_lengths):.0f} tokens")
            print(f"  Std:    {token_lengths.std():.0f} tokens")
            print(f"  Min:    {token_lengths.min():.0f} tokens")
            print(f"  Max:    {token_lengths.max():.0f} tokens")
            
            # Percentiles
            print(f"\nPercentiles:")
            for p in [25, 50, 75, 90, 95, 99]:
                val = np.percentile(token_lengths, p)
                print(f"  {p}th: {val:.0f} tokens")
            
            # How many exceed model limits?
            over_limit = (token_lengths > max_tokens).sum()
            over_pct = (over_limit / len(token_lengths)) * 100
            
            print(f"\n⚠️  Reviews exceeding {max_tokens} token limit:")
            print(f"  Count: {over_limit:,} / {len(token_lengths):,} ({over_pct:.1f}%)")
            print(f"  These will be truncated during embedding!")
            
            # Average truncation loss
            truncated_tokens = token_lengths[token_lengths > max_tokens]
            if len(truncated_tokens) > 0:
                avg_loss = (truncated_tokens - max_tokens).mean()
                max_loss = (truncated_tokens - max_tokens).max()
                print(f"\n  For truncated reviews:")
                print(f"    Avg tokens lost: {avg_loss:.0f} ({avg_loss/truncated_tokens.mean()*100:.1f}%)")
                print(f"    Max tokens lost: {max_loss:.0f}")
            
            # Distribution
            print(f"\nToken Length Distribution:")
            bins = [0, 64, 128, 256, 384, 512, 768, 1024, 10000]
            labels = ['0-64', '64-128', '128-256', '256-384', '384-512', '512-768', '768-1024', '1024+']
            token_bins = pd.cut(token_lengths, bins=bins, labels=labels)
            token_dist = pd.Series(token_bins).value_counts().sort_index()
            
            for label, count in token_dist.items():
                pct = (count / len(token_lengths)) * 100
                marker = " ← Fits" if int(label.split('-')[1].replace('+', '')) <= max_tokens else " ← TRUNCATED"
                print(f"  {label:12s}: {count:6,} ({pct:5.1f}%){marker if pct > 0 else ''}")
            
            results[model_name] = {
                'mean': token_lengths.mean(),
                'median': np.median(token_lengths),
                'over_limit': over_pct,
                'max_tokens': max_tokens
            }
            
        except Exception as e:
            print(f"  ❌ Error loading {model_name}: {e}")
            continue
    
    # Summary comparison
    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)
    
    print(f"\n{'Model':<20} {'Max Tokens':<12} {'% Truncated':<15} {'Coverage'}")
    print("-" * 80)
    for model_name, stats in results.items():
        coverage = 100 - stats['over_limit']
        bar = "█" * int(coverage / 5) + "░" * (20 - int(coverage / 5))
        print(f"{model_name:<20} {stats['max_tokens']:<12} {stats['over_limit']:>6.1f}%        {bar} {coverage:.1f}%")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    if 'all-MiniLM-L6-v2' in results:
        miniLM_truncated = results['all-MiniLM-L6-v2']['over_limit']
        
        if miniLM_truncated > 30:
            print("\n⚠️  WARNING: Current model (MiniLM-L6-v2) will truncate {:.1f}% of reviews!".format(miniLM_truncated))
            print("\n   Recommended actions:")
            print("   1. Switch to bge-base-en-v1.5 (512 tokens) - best balance")
            print("   2. Or add truncation to 900 chars BEFORE adding business name")
            print("   3. Or use all-mpnet-base-v2 (384 tokens) as middle ground")
        elif miniLM_truncated > 15:
            print("\n⚠️  MODERATE: {:.1f}% of reviews will be truncated with current model".format(miniLM_truncated))
            print("\n   Suggested improvements:")
            print("   1. Add truncation to 900 chars before adding business name")
            print("   2. Or switch to bge-base-en-v1.5 for better coverage")
        else:
            print("\n✅ GOOD: Current model covers {:.1f}% of reviews without truncation".format(100 - miniLM_truncated))
            print("   No urgent action needed, but consider bge-base for slight improvement")
    
    # Check if business names are already added
    sample_text = df['text'].iloc[0]
    if sample_text.startswith("Business:"):
        print("\n📌 NOTE: Reviews already have 'Business:' prefix")
        print("   Business names add ~8-15 tokens per review")
        print("   Consider this when setting truncation limits")
    else:
        print("\n📌 NOTE: Business names NOT yet added to reviews")
        print("   After adding 'Business: [name]. Review:' prefix:")
        print("   - Expect ~8-15 additional tokens per review")
        print("   - Truncate reviews to 900 chars BEFORE adding prefix (for 256 limit)")
        print("   - Truncate reviews to 1900 chars BEFORE adding prefix (for 512 limit)")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    
    return df, results


if __name__ == "__main__":
    import sys
    
    csv_path = 'reviews_for_embeddings.csv'
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    
    df, results = analyze_review_lengths(csv_path)
    
    print("\n💡 TIP: Run with custom CSV path:")
    print("   python analyze_review_lengths.py path/to/your/reviews.csv")
