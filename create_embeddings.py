"""
Script to create embeddings for Yelp reviews and store them in a FAISS index.
"""

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import Optional, Tuple
import pickle
from tqdm import tqdm
import json
from sklearn.model_selection import train_test_split


class ReviewEmbeddingsCreator:
    """Create and manage embeddings for Yelp reviews using FAISS."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the embeddings creator.
        
        Args:
            model_name: Name of the sentence transformer model to use
                       'all-MiniLM-L6-v2' is fast and produces 384-dim embeddings
                       'all-mpnet-base-v2' is slower but higher quality, 768-dim
        """
        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.review_metadata = None
        self.train_df = None
        self.holdout_df = None
    
    def create_train_holdout_split(self,
                                   reviews_df: pd.DataFrame,
                                   holdout_size: float = 0.2,
                                   random_state: int = 42,
                                   stratify_column: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split reviews into training and holdout sets.
        
        Args:
            reviews_df: DataFrame containing all reviews
            holdout_size: Fraction of data to use for holdout (default 0.2 = 20%)
            random_state: Random seed for reproducibility
            stratify_column: Optional column name to stratify split (e.g., 'stars' for rating balance)
            
        Returns:
            Tuple of (train_df, holdout_df)
        """
        print(f"\nCreating train/holdout split...")
        print(f"Total reviews: {len(reviews_df)}")
        print(f"Holdout size: {holdout_size:.1%}")
        
        stratify = reviews_df[stratify_column] if stratify_column else None
        
        train_df, holdout_df = train_test_split(
            reviews_df,
            test_size=holdout_size,
            random_state=random_state,
            stratify=stratify
        )
        
        # Reset indices to avoid confusion
        train_df = train_df.reset_index(drop=True)
        holdout_df = holdout_df.reset_index(drop=True)
        
        print(f"Training set: {len(train_df)} reviews ({len(train_df)/len(reviews_df):.1%})")
        print(f"Holdout set: {len(holdout_df)} reviews ({len(holdout_df)/len(reviews_df):.1%})")
        
        if stratify_column:
            print(f"\nTraining set {stratify_column} distribution:")
            print(train_df[stratify_column].value_counts().sort_index())
            print(f"\nHoldout set {stratify_column} distribution:")
            print(holdout_df[stratify_column].value_counts().sort_index())
        
        # Store the splits
        self.train_df = train_df
        self.holdout_df = holdout_df
        
        return train_df, holdout_df
        
    def create_embeddings(self, 
                         reviews_df: pd.DataFrame,
                         text_column: str = 'text',
                         batch_size: int = 32,
                         show_progress: bool = True) -> np.ndarray:
        """
        Create embeddings for all reviews.
        
        Args:
            reviews_df: DataFrame containing reviews
            text_column: Name of the column containing review text
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            
        Returns:
            numpy array of embeddings (n_reviews x embedding_dim)
        """
        texts = reviews_df[text_column].tolist()
        
        print(f"Creating embeddings for {len(texts)} reviews...")
        print(f"Embedding dimension: {self.embedding_dim}")
        
        # Create embeddings in batches
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def build_faiss_index(self, 
                         embeddings: np.ndarray,
                         use_gpu: bool = False,
                         index_type: str = 'flat') -> faiss.Index:
        """
        Build a FAISS index from embeddings using cosine similarity.
        
        Args:
            embeddings: numpy array of embeddings
            use_gpu: Whether to use GPU for indexing (requires faiss-gpu)
            index_type: Type of index to build
                       'flat' - exact search, slower but most accurate
                       'ivf' - approximate search, faster
                       
        Returns:
            FAISS index
        """
        n_embeddings = embeddings.shape[0]
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        if index_type == 'flat':
            # Use Inner Product for normalized vectors = cosine similarity
            index = faiss.IndexFlatIP(self.embedding_dim)
            
        elif index_type == 'ivf':
            # Approximate search using IVF (Inverted File Index)
            n_clusters = min(int(np.sqrt(n_embeddings)), 1000)
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, n_clusters)
            
            # Train the index
            print(f"Training IVF index with {n_clusters} clusters...")
            index.train(embeddings)
        
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # Add embeddings to index
        print(f"Adding {n_embeddings} embeddings to index...")
        index.add(embeddings)
        
        if use_gpu:
            # Move index to GPU
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        
        self.index = index
        print(f"Index built successfully. Total vectors: {index.ntotal}")
        
        return index
    
    def search(self, 
               query: str,
               k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """
        Search for similar reviews using cosine similarity.
        
        Args:
            query: Query text
            k: Number of nearest neighbors to return
            
        Returns:
            Tuple of (cosine_similarities, indices)
        """
        if self.index is None:
            raise ValueError("Index not built yet. Call build_faiss_index() first.")
        
        # Encode and normalize query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search - returns cosine similarities
        similarities, indices = self.index.search(query_embedding, k)
        
        return similarities[0], indices[0]
    
    def save(self, 
             index_path: str = 'faiss_index.bin',
             metadata_path: str = 'review_metadata.pkl',
             train_path: Optional[str] = None,
             holdout_path: Optional[str] = None):
        """
        Save FAISS index and metadata to disk.
        
        Args:
            index_path: Path to save FAISS index
            metadata_path: Path to save metadata (review IDs, etc.)
            train_path: Optional path to save training DataFrame
            holdout_path: Optional path to save holdout DataFrame
        """
        if self.index is None:
            raise ValueError("No index to save. Call build_faiss_index() first.")
        
        # Save FAISS index
        faiss.write_index(self.index, index_path)
        print(f"Saved FAISS index to {index_path}")
        
        # Save metadata if available
        if self.review_metadata is not None:
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.review_metadata, f)
            print(f"Saved metadata to {metadata_path}")
        
        # Save train/holdout splits if available
        if train_path and self.train_df is not None:
            self.train_df.to_csv(train_path, index=False)
            print(f"Saved training set to {train_path}")
        
        if holdout_path and self.holdout_df is not None:
            self.holdout_df.to_csv(holdout_path, index=False)
            print(f"Saved holdout set to {holdout_path}")
    
    def load(self, 
             index_path: str = 'faiss_index.bin',
             metadata_path: str = 'review_metadata.pkl',
             train_path: Optional[str] = None,
             holdout_path: Optional[str] = None):
        """
        Load FAISS index and metadata from disk.
        
        Args:
            index_path: Path to FAISS index
            metadata_path: Path to metadata file
            train_path: Optional path to training DataFrame
            holdout_path: Optional path to holdout DataFrame
        """
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        print(f"Loaded FAISS index from {index_path}")
        print(f"Index contains {self.index.ntotal} vectors")
        
        # Load metadata if available
        try:
            with open(metadata_path, 'rb') as f:
                self.review_metadata = pickle.load(f)
            print(f"Loaded metadata from {metadata_path}")
        except FileNotFoundError:
            print(f"No metadata file found at {metadata_path}")
        
        # Load train/holdout splits if available
        if train_path:
            try:
                self.train_df = pd.read_csv(train_path)
                print(f"Loaded training set from {train_path} ({len(self.train_df)} reviews)")
            except FileNotFoundError:
                print(f"No training set file found at {train_path}")
        
        if holdout_path:
            try:
                self.holdout_df = pd.read_csv(holdout_path)
                print(f"Loaded holdout set from {holdout_path} ({len(self.holdout_df)} reviews)")
            except FileNotFoundError:
                print(f"No holdout set file found at {holdout_path}")
    
    def get_similar_reviews(self,
                           query: str,
                           reviews_df: pd.DataFrame,
                           k: int = 5) -> pd.DataFrame:
        """
        Get similar reviews with full metadata.
        
        Args:
            query: Query text
            reviews_df: Original reviews DataFrame
            k: Number of results to return
            
        Returns:
            DataFrame with similar reviews and cosine similarities
        """
        similarities, indices = self.search(query, k)
        
        # Get reviews at those indices
        similar_reviews = reviews_df.iloc[indices].copy()
        similar_reviews['cosine_similarity'] = similarities
        
        return similar_reviews
    
    def search_and_save_json(self,
                            query: str,
                            reviews_df: pd.DataFrame,
                            output_path: str,
                            k: int = 10,
                            similarity_threshold: float = 0.7) -> dict:
        """
        Search for similar reviews and save results to JSON file.
        
        Args:
            query: Query text
            reviews_df: Original reviews DataFrame
            output_path: Path to save JSON output
            k: Number of results to retrieve (before filtering)
            similarity_threshold: Minimum cosine similarity to include (default 0.7)
            
        Returns:
            Dictionary with query results
        """
        # Get similar reviews
        similar_reviews = self.get_similar_reviews(query, reviews_df, k=k)
        
        # Filter by similarity threshold
        filtered_reviews = similar_reviews[similar_reviews['cosine_similarity'] >= similarity_threshold]
        
        # Build output structure
        output = {
            'query': query,
            'similarity_threshold': similarity_threshold,
            'total_retrieved': len(similar_reviews),
            'total_above_threshold': len(filtered_reviews),
            'reviews': []
        }
        
        # Add each review
        for idx, row in filtered_reviews.iterrows():
            review_data = {
                'review_id': row['review_id'],
                'business_id': row['business_id'],
                'stars': float(row['stars']),
                'cosine_similarity': float(row['cosine_similarity']),
                'text': row['text'],
                'useful': int(row['useful']) if 'useful' in row else 0,
                'funny': int(row['funny']) if 'funny' in row else 0,
                'cool': int(row['cool']) if 'cool' in row else 0
            }
            output['reviews'].append(review_data)
        
        # Save to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(filtered_reviews)} reviews (similarity >= {similarity_threshold}) to {output_path}")
        
        return output
