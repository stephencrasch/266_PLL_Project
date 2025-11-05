"""
Create train/holdout split and FAISS index from reviews_for_embeddings.csv
Run this ONCE before running full_run.py for the first time.
"""

import pandas as pd
from embeddings import ReviewEmbeddingsCreator


def main():
    print("=" * 80)
    print("Creating Train/Holdout Split and FAISS Index")
    print("=" * 80)
    
    # Initialize embeddings creator
    creator = ReviewEmbeddingsCreator(model_name='all-MiniLM-L6-v2')
    
    # Load all reviews
    print("\nLoading reviews...")
    reviews_df = pd.read_csv('reviews_for_embeddings.csv')
    print(f"Loaded {len(reviews_df)} total reviews")
    print(f"Label distribution:\n{reviews_df['label'].value_counts()}")
    
    # Create train/holdout split (stratified by label)
    train_df, holdout_df = creator.create_train_holdout_split(
        reviews_df,
        holdout_size=0.2,  # 20% holdout
        random_state=42,
        stratify_column='label'  # Keep same label distribution in both sets
    )
    
    # Create embeddings for TRAINING SET ONLY
    print("\n" + "=" * 80)
    print("Creating embeddings for TRAINING SET")
    print("=" * 80)
    train_embeddings = creator.create_embeddings(
        train_df,
        text_column='text',
        batch_size=32,
        show_progress=True
    )
    
    # Build FAISS index from training embeddings
    print("\n" + "=" * 80)
    print("Building FAISS Index")
    print("=" * 80)
    creator.build_faiss_index(
        train_embeddings,
        use_gpu=False,
        index_type='flat'
    )
    
    # Save everything
    print("\n" + "=" * 80)
    print("Saving Files")
    print("=" * 80)
    creator.save(
        index_path='review_embeddings_train.faiss',
        metadata_path='review_metadata.pkl',
        train_path='reviews_train.csv',
        holdout_path='reviews_holdout.csv'
    )
    
    print("\n" + "=" * 80)
    print("✅ COMPLETE!")
    print("=" * 80)
    print(f"Training set: {len(train_df)} reviews (in FAISS index)")
    print(f"Holdout set: {len(holdout_df)} reviews (NOT in FAISS index)")
    print(f"\nFiles created:")
    print(f"  • review_embeddings_train.faiss  - FAISS index with training embeddings")
    print(f"  • review_metadata.pkl            - Metadata")
    print(f"  • reviews_train.csv              - Training reviews")
    print(f"  • reviews_holdout.csv            - Holdout reviews (for final evaluation)")
    print(f"\n✅ You can now run full_run.py for Round 0!")
    print("=" * 80)


if __name__ == '__main__':
    main()
