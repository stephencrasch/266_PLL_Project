"""
Diagnostic script to understand why duplicate retrievals occur with model filtering.
"""

import pandas as pd
import json
import os

def diagnose_duplicates():
    """Analyze duplicate retrievals across categories."""
    
    print("=" * 80)
    print("DUPLICATE RETRIEVAL DIAGNOSTIC")
    print("=" * 80)
    
    # Load the latest round results
    current_round = 1
    json_file = f"all_category_results_r{current_round}.json"
    predictions_file = f"predictions_r{current_round - 1}.csv"
    
    if not os.path.exists(json_file):
        print(f"❌ {json_file} not found. Run full_run.py first.")
        return
    
    print(f"\n📂 Loading {json_file}...")
    with open(json_file, 'r') as f:
        results = json.load(f)
    
    # Build dataframe of all retrievals
    rows = []
    for cat in results['categories']:
        category = cat['category']
        for review in cat['reviews']:
            rows.append({
                'review_id': review['review_id'],
                'category': category,
                'ce_score': review['ce_score'],
                'text': review['text'][:200]  # First 200 chars
            })
    
    df = pd.DataFrame(rows)
    print(f"\n✅ Loaded {len(df)} total retrievals across {len(results['categories'])} categories")
    
    # Find duplicates
    duplicates = df[df.duplicated(subset=['review_id'], keep=False)]
    unique_dup_reviews = duplicates['review_id'].unique()
    
    print(f"\n{'='*80}")
    print(f"DUPLICATE SUMMARY")
    print(f"{'='*80}")
    print(f"Total retrievals: {len(df)}")
    print(f"Unique reviews: {df['review_id'].nunique()}")
    print(f"Duplicate reviews: {len(unique_dup_reviews)} ({len(unique_dup_reviews)/df['review_id'].nunique()*100:.1f}%)")
    print(f"Total duplicate instances: {len(duplicates)} ({len(duplicates)/len(df)*100:.1f}%)")
    
    if len(unique_dup_reviews) == 0:
        print("\n✅ No duplicates found! Model filtering is working perfectly.")
        return
    
    # Analyze which categories have overlaps
    print(f"\n{'='*80}")
    print(f"CATEGORY OVERLAP ANALYSIS")
    print(f"{'='*80}")
    
    dup_pairs = {}
    for review_id in unique_dup_reviews:
        review_cats = df[df['review_id'] == review_id]['category'].tolist()
        cat_pair = tuple(sorted(review_cats))
        dup_pairs[cat_pair] = dup_pairs.get(cat_pair, 0) + 1
    
    print("\nMost common category pairs:")
    for pair, count in sorted(dup_pairs.items(), key=lambda x: x[1], reverse=True):
        print(f"  {' ↔ '.join(pair)}: {count} reviews")
    
    # Check if predictions file exists
    if not os.path.exists(predictions_file):
        print(f"\n⚠️  {predictions_file} not found. Cannot analyze model predictions.")
        print(f"    This is expected for Round 0 (no previous model).")
        return
    
    print(f"\n{'='*80}")
    print(f"MODEL PREDICTION ANALYSIS")
    print(f"{'='*80}")
    
    print(f"\n📂 Loading {predictions_file}...")
    pred_df = pd.read_csv(predictions_file)
    
    # Check for duplicate review_ids in predictions
    if pred_df['review_id'].duplicated().any():
        dup_count = pred_df['review_id'].duplicated().sum()
        print(f"\n⚠️  WARNING: predictions file has {dup_count} duplicate review_ids!")
        print(f"    Each review should appear only once in predictions.")
        print(f"\nExample duplicates:")
        dup_ids = pred_df[pred_df['review_id'].duplicated(keep=False)]['review_id'].unique()[:5]
        for rid in dup_ids:
            dup_rows = pred_df[pred_df['review_id'] == rid]
            print(f"\n  Review {rid}:")
            print(dup_rows[['review_id', 'predicted_label', 'confidence']])
    else:
        print(f"✅ No duplicate predictions (each review has 1 prediction)")
    
    # Analyze predictions for duplicate reviews
    print(f"\n{'='*80}")
    print(f"PREDICTIONS FOR DUPLICATE REVIEWS")
    print(f"{'='*80}")
    
    # Get predictions for duplicated reviews
    dup_preds = pred_df[pred_df['review_id'].isin(unique_dup_reviews)]
    
    if len(dup_preds) == 0:
        print("\n❌ None of the duplicate reviews found in predictions file!")
        print("   This suggests review_ids don't match between retrieval and predictions.")
        return
    
    print(f"\nFound predictions for {len(dup_preds)}/{len(unique_dup_reviews)} duplicate reviews")
    
    # Analyze top 10 duplicates in detail
    print(f"\n{'='*80}")
    print(f"TOP 10 DUPLICATE REVIEWS (DETAILED)")
    print(f"{'='*80}")
    
    for idx, review_id in enumerate(unique_dup_reviews[:10], 1):
        print(f"\n{'─'*80}")
        print(f"Duplicate #{idx}: Review ID {review_id}")
        print(f"{'─'*80}")
        
        # Get retrieval info
        review_data = df[df['review_id'] == review_id]
        print(f"\nRetrieved by {len(review_data)} categories:")
        for _, row in review_data.iterrows():
            print(f"  • {row['category']}: CE score = {row['ce_score']:.4f}")
        
        # Get prediction info
        pred_row = pred_df[pred_df['review_id'] == review_id]
        if len(pred_row) > 0:
            pred_row = pred_row.iloc[0]
            print(f"\nModel Prediction (Round {current_round-1}):")
            print(f"  • Predicted: {pred_row['predicted_label']}")
            print(f"  • Confidence: {pred_row['confidence']:.4f}")
            
            # Show all probabilities
            prob_cols = [c for c in pred_df.columns if c.startswith('prob_')]
            if prob_cols:
                print(f"\n  Probabilities for all categories:")
                for col in sorted(prob_cols):
                    cat_name = col.replace('prob_', '')
                    prob = pred_row[col]
                    marker = " ✓" if cat_name == pred_row['predicted_label'] else ""
                    threshold_marker = " [≥0.5]" if prob >= 0.5 else ""
                    print(f"    {cat_name}: {prob:.4f}{marker}{threshold_marker}")
            
            # Check which categories this review SHOULD pass filtering for
            categories_above_threshold = []
            for col in prob_cols:
                cat_name = col.replace('prob_', '')
                if pred_row[col] >= 0.5 and cat_name == pred_row['predicted_label']:
                    categories_above_threshold.append(cat_name)
            
            print(f"\n  Should pass filtering for: {categories_above_threshold if categories_above_threshold else 'NONE (conf < 0.5 or not winner)'}")
            
            # Check if retrieved categories match expected filtering
            retrieved_cats = review_data['category'].tolist()
            print(f"  Actually retrieved by: {retrieved_cats}")
            
            unexpected = [c for c in retrieved_cats if c not in categories_above_threshold]
            if unexpected:
                print(f"  ⚠️  UNEXPECTED: Retrieved by {unexpected} despite prediction!")
        else:
            print(f"\n❌ No prediction found for this review")
        
        # Show text preview
        text = review_data.iloc[0]['text']
        print(f"\nText preview:")
        print(f"  {text[:300]}...")
    
    # Summary statistics
    print(f"\n{'='*80}")
    print(f"FILTERING EFFECTIVENESS ANALYSIS")
    print(f"{'='*80}")
    
    # Check if filtering was actually applied
    for cat_result in results['categories']:
        category = cat_result['category']
        total_retrieved = cat_result.get('total_after_merging', 0)
        filtered_count = cat_result.get('filtered_doc_count', total_retrieved)
        used_filtering = cat_result.get('used_model_filtering', False)
        
        if used_filtering:
            if filtered_count < total_retrieved:
                pct = (filtered_count / total_retrieved * 100) if total_retrieved > 0 else 0
                print(f"\n{category}:")
                print(f"  Retrieved: {total_retrieved} docs")
                print(f"  After filtering: {filtered_count} docs ({pct:.1f}%)")
                print(f"  Removed: {total_retrieved - filtered_count} docs")
            else:
                print(f"\n{category}:")
                print(f"  Retrieved: {total_retrieved} docs")
                print(f"  ⚠️  Filtering had NO EFFECT (all docs kept)")
        else:
            print(f"\n{category}:")
            print(f"  Retrieved: {total_retrieved} docs")
            print(f"  ❌ Filtering was NOT APPLIED")
    
    # Final recommendations
    print(f"\n{'='*80}")
    print(f"DIAGNOSIS & RECOMMENDATIONS")
    print(f"{'='*80}")
    
    # Calculate how many duplicates have prob >= 0.5 for multiple categories
    multi_category_high_conf = 0
    for review_id in unique_dup_reviews:
        pred_row = pred_df[pred_df['review_id'] == review_id]
        if len(pred_row) > 0:
            pred_row = pred_row.iloc[0]
            prob_cols = [c for c in pred_df.columns if c.startswith('prob_')]
            high_conf_cats = sum(1 for col in prob_cols if pred_row[col] >= 0.5)
            if high_conf_cats > 1:
                multi_category_high_conf += 1
    
    print(f"\n🔍 Key Findings:")
    print(f"  • {len(unique_dup_reviews)} reviews retrieved by multiple categories")
    print(f"  • {multi_category_high_conf} of these have prob ≥ 0.5 for multiple categories")
    
    if multi_category_high_conf > 0:
        print(f"\n✅ Expected Behavior:")
        print(f"  Model predictions show {multi_category_high_conf} reviews are genuinely ambiguous")
        print(f"  These reviews legitimately belong to multiple categories")
        print(f"  Filtering uses argmax (winner-take-all), so they should only pass ONE filter")
        print(f"\n⚠️  If duplicates persist, check:")
        print(f"  1. Is filtering logic checking argmax correctly?")
        print(f"  2. Is min_docs_for_tfidf fallback triggering? (bypasses filtering)")
        print(f"  3. Are review_ids matching between retrieval and predictions?")
    else:
        print(f"\n❌ Unexpected Behavior:")
        print(f"  Duplicates exist but model doesn't show high confidence for multiple categories")
        print(f"  This suggests filtering is not being applied correctly")
        print(f"\n🔧 Check:")
        print(f"  1. Is use_model_filtering=True in config?")
        print(f"  2. Does predictions file have correct review_ids?")
        print(f"  3. Is filtering logic comparing predicted_label correctly?")
    
    print(f"\n{'='*80}")


if __name__ == "__main__":
    diagnose_duplicates()
