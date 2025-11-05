"""
Example script showing how to create embeddings with a train/holdout split.
The FAISS index will only contain the training set, keeping the holdout set separate for evaluation.
"""

import pandas as pd
from embeddings import ReviewEmbeddingsCreator


def main():
    """
    Create embeddings with train/holdout split.
    """
    
    # Initialize the embeddings creator
    print("=" * 80)
    print("Creating Review Embeddings with Train/Holdout Split")
    print("=" * 80)
    
    creator = ReviewEmbeddingsCreator(model_name='all-MiniLM-L6-v2')
    
    # Load your reviews data
    print("\nLoading reviews data...")
    reviews_df = pd.read_csv('reviews_for_embeddings.csv')
    print(f"Loaded {len(reviews_df)} total reviews")
    
    # Create train/holdout split
    # Using stratify_column='stars' ensures both sets have similar rating distributions
    train_df, holdout_df = creator.create_train_holdout_split(
        reviews_df,
        holdout_size=0.2,  # 20% for holdout
        random_state=42,
        stratify_column='stars'  # Stratify by star rating
    )
    
    # Create embeddings for TRAINING SET ONLY
    print("\n" + "=" * 80)
    print("Creating embeddings for TRAINING SET only")
    print("=" * 80)
    train_embeddings = creator.create_embeddings(
        train_df,
        text_column='text',
        batch_size=32,
        show_progress=True
    )
    
    # Build FAISS index using TRAINING SET ONLY
    print("\n" + "=" * 80)
    print("Building FAISS index from TRAINING SET")
    print("=" * 80)
    creator.build_faiss_index(
        train_embeddings,
        use_gpu=False,
        index_type='flat'
    )
    
    # Save everything
    print("\n" + "=" * 80)
    print("Saving index and data splits")
    print("=" * 80)
    creator.save(
        index_path='review_embeddings_train.faiss',
        metadata_path='review_metadata_train.pkl',
        train_path='reviews_train.csv',
        holdout_path='reviews_holdout.csv'
    )
    
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"✓ Training set: {len(train_df)} reviews → FAISS index")
    print(f"✓ Holdout set: {len(holdout_df)} reviews → Saved separately (NOT in index)")
    print(f"✓ FAISS index contains: {creator.index.ntotal} vectors (training set only)")
    print(f"\nFiles created:")
    print(f"  - review_embeddings_train.faiss  (FAISS index with training embeddings)")
    print(f"  - review_metadata_train.pkl       (metadata)")
    print(f"  - reviews_train.csv               (training reviews)")
    print(f"  - reviews_holdout.csv             (holdout reviews for evaluation)")
    print("\n" + "=" * 80)
    
    # Optional: Create embeddings for holdout set (for evaluation)
    print("\nCreating embeddings for HOLDOUT SET (for evaluation)...")
    holdout_embeddings = creator.create_embeddings(
        holdout_df,
        text_column='text',
        batch_size=32,
        show_progress=True
    )
    
    # Save holdout embeddings separately (NOT added to index)
    import numpy as np
    np.save('holdout_embeddings.npy', holdout_embeddings)
    print(f"Saved holdout embeddings to holdout_embeddings.npy")
    print(f"Shape: {holdout_embeddings.shape}")
    
    print("\n" + "=" * 80)
    print("Complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Use the training FAISS index for retrieval")
    print("2. Use the holdout set for evaluation/testing")
    print("3. Ensure you never search the holdout reviews during training/development")
    

if __name__ == '__main__':
    main()
