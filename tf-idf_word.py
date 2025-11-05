"""
Script to calculate TF-IDF scores for retrieved reviews and append to search results JSON.
Includes cross-encoder reranking for improved ranking quality with category specificity.
"""

import os
# Fix multiprocessing issues on M3 Mac
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import CrossEncoder
from typing import List, Dict, Tuple, Optional
import torch


def choose_device(prefer=None):
    """Choose the best available device for cross-encoder."""
    if prefer:
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    # MPS available on recent PyTorch builds for Apple silicon
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# Import NLTK for lemmatization
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data (will skip if already downloaded)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()


def lemmatize_text(text: str) -> str:
    """Lemmatize text by tokenizing and lemmatizing each word."""
    tokens = word_tokenize(text.lower())
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha()]
    return ' '.join(lemmatized)

# Generic words to exclude from query expansion
GENERIC_BLACKLIST = {
    # Sentiment adjectives
    'great', 'good', 'bad', 'friendly', 'best', 'excellent', 'amazing', 'awesome', 'terrible',
    'nice', 'wonderful', 'fantastic', 'bully', 'horrible', 'awful', 'decent, like', 'franklin',
    
    # Generic location/service words
    'place', 'spot', 'location', 'area', 'service', 'staff', 'experience',
    'time', 'times', 'visit', 'definitely', 'highly', 'really', 'always',
    
    # Generic food/drink (too broad)
    'food', 'drinks', 'drink', 'menu', 'order', 'ordered',
    
    # Recommendation language
    'recommend', 'lot', 'recommended', 'love', 'just', 'loved', 'favorite', 'like', 'best', 'worst',
    
    # Filler words
    'got', 'get', 'went', 'go', 'come', 'came', 'back', 'try', 'tried'
}


def load_search_results(json_path: str) -> dict:
    """
    Load search results from JSON or JSONL file.
    
    Args:
        json_path: Path to JSON or JSONL file
        
    Returns:
        Dictionary with search results
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        # Try JSON first
        try:
            f.seek(0)
            return json.load(f)
        except json.JSONDecodeError:
            # Try JSONL format
            f.seek(0)
            lines = f.readlines()
            reviews = [json.loads(line) for line in lines if line.strip()]
            
            # Convert JSONL to expected format
            return {
                'query': 'unknown',
                'total_above_threshold': len(reviews),
                'reviews': reviews
            }


def calculate_tfidf(texts: List[str], top_n: int = 20, max_features: int = 1000) -> Dict:
    """
    Calculate TF-IDF scores for a collection of texts with lemmatization.
    Returns fewer than top_n if vocabulary is smaller.
    """
    if not texts:
        return {
            'top_words': [],
            'total_unique_terms': 0,
            'total_documents': 0
        }
    
    # Lemmatize all texts first
    print(f"  Lemmatizing {len(texts)} documents...")
    lemmatized_texts = [lemmatize_text(text) for text in texts]
    
    # Adjust min_df for small document sets
    min_df = 1 if len(texts) < 5 else 2
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        ngram_range=(1,1),  # Allow only unigrams
        min_df=min_df,
        max_df=0.8,
        lowercase=True
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(lemmatized_texts)
        feature_names = vectorizer.get_feature_names_out()
        
        # Check if we got any features
        if len(feature_names) == 0:
            return {
                'top_words': [],
                'total_unique_terms': 0,
                'total_documents': len(texts)
            }
        
        avg_tfidf_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
        
        # Get top N terms (or fewer if vocabulary is smaller)
        actual_top_n = min(top_n, len(feature_names))
        top_indices = avg_tfidf_scores.argsort()[-actual_top_n:][::-1]
        
        top_words = [
            {
                'word': feature_names[idx],
                'tfidf_score': float(avg_tfidf_scores[idx])
            }
            for idx in top_indices
        ]
        
        return {
            'top_words': top_words,
            'total_unique_terms': len(feature_names),
            'total_documents': len(texts)
        }
        
    except Exception as e:
        print(f"  ⚠ TF-IDF calculation failed: {e}")
        return {
            'top_words': [],
            'total_unique_terms': 0,
            'total_documents': len(texts)
        }


class QueryExpander:
    """Class to handle query expansion with TF-IDF + category specificity."""
    
    def __init__(self, global_corpus: Optional[List[str]] = None):
        """
        Initialize query expander.
        
        Args:
            global_corpus: Optional list of all review texts for global TF-IDF
        """
        # Store lemmatizer
        self.lemmatizer = lemmatizer
        print("Query expander initialized")
        
        # Pre-compute global TF-IDF for category specificity
        self.global_vocab_scores = {}
        self.global_corpus = None  # Will store LEMMATIZED version
        
        if global_corpus and len(global_corpus) > 0:
            print(f"Computing global TF-IDF on {len(global_corpus)} documents...")
            print(f"  Lemmatizing global corpus...")
            
            # Lemmatize global corpus and STORE IT (critical for specificity calculation!)
            lemmatized_global = [lemmatize_text(text) for text in global_corpus]
            self.global_corpus = lemmatized_global  # Store LEMMATIZED version!
            
            min_df_global = 1 if len(global_corpus) < 5 else 2
            
            vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 1),  # Allow unigrams and bigrams
                min_df=min_df_global,
                max_df=0.8,
                lowercase=True
            )
            
            global_matrix = vectorizer.fit_transform(lemmatized_global)
            feature_names = vectorizer.get_feature_names_out()
            avg_scores = np.asarray(global_matrix.mean(axis=0)).flatten()
            
            self.global_vocab_scores = {
                feature_names[i]: float(avg_scores[i])
                for i in range(len(feature_names))
            }
            print(f"  Global vocabulary: {len(self.global_vocab_scores)} terms")
    
    def calculate_category_specificity(self, word: str, local_texts: List[str]) -> float:
        """
        Calculate how category-specific a word is using document frequency.
        
        Specificity = (local_doc_freq / local_total) / (global_doc_freq / global_total)
        High score = word appears more frequently in category docs than in global corpus
        
        Args:
            word: The candidate word
            local_texts: List of category-specific texts
            
        Returns:
            Category specificity score (ratio)
        """
        if not self.global_corpus:
            return 0.0
        
        word_lower = word.lower()
        
        # Count documents containing the word in local corpus
        local_doc_freq = sum(1 for text in local_texts if word_lower in text.lower())
        local_total = len(local_texts)
        
        # Count documents containing the word in global corpus
        global_doc_freq = sum(1 for text in self.global_corpus if word_lower in text.lower())
        global_total = len(self.global_corpus)
        
        # Calculate rates
        local_rate = local_doc_freq / local_total if local_total > 0 else 0.0
        global_rate = global_doc_freq / global_total if global_total > 0 else 0.0
        
        # Specificity ratio with epsilon to avoid division by zero
        epsilon = 1e-6
        specificity = local_rate / (global_rate + epsilon)
        
        # Log transform for better scale
        log_specificity = np.log1p(specificity)
        
        # Length adjustment: penalize longer n-grams (they're artificially rarer)
        # This ensures fair competition between unigrams and bigrams
        n_gram_length = len(word.split())
        length_penalty = (n_gram_length - 1) * 0.5  # 0.5 penalty per additional word
        
        adjusted_specificity = log_specificity - length_penalty
        
        return float(adjusted_specificity)
    
    def normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Normalize scores using min-max normalization with clipping.
        
        Args:
            scores: Array of scores
            
        Returns:
            Normalized scores in [0, 1]
        """
        scores = np.array(scores)
        if len(scores) == 0:
            return scores
        
        # Handle all zeros or all same
        if scores.max() == scores.min():
            if scores.max() == 0.0:
                return np.zeros_like(scores)
            return np.full_like(scores, 0.5)
        
        normalized = (scores - scores.min()) / (scores.max() - scores.min())
        return np.clip(normalized, 0.001, 0.999)
    
    def filter_candidates(self, candidates: List[Dict], query: str) -> List[Dict]:
        """
        Filter candidates with improved logic.
        - Remove generic blacklisted words
        - Remove exact token matches and lemmas
        - Allow bigrams that add new information
        - Remove invalid tokens (digits, too short)
        
        Note: Candidate words from TF-IDF are already lemmatized by our custom tokenizer
        """
        query_lower = query.lower()
        query_tokens = set(query_lower.split())
        
        # Get lemmatized forms of query tokens
        if self.lemmatizer:
            query_lemmas = set(self.lemmatizer.lemmatize(t) for t in query_tokens)
        else:
            query_lemmas = set()
        
        filtered = []
        for candidate in candidates:
            word = candidate['word']
            word_lower = word.lower()
            
            # Skip if in generic blacklist (check first)
            if word_lower in GENERIC_BLACKLIST:
                continue
            
            # Skip invalid candidates
            if len(word) < 3 or any(c.isdigit() for c in word):
                continue
            
            # Skip exact token match (lowercase)
            if word_lower in query_tokens:
                continue
            
            # Skip if candidate matches any lemmatized query token
            # (Candidate is already lemmatized, so direct comparison)
            if word_lower in query_lemmas:
                continue
            
            # For multi-word candidates (bigrams), check if all parts are in query
            word_tokens = word_lower.split()
            if len(word_tokens) > 1:
                # Skip if all tokens are already in original query
                if set(word_tokens).issubset(query_tokens):
                    continue
                # Skip if all tokens match lemmatized query tokens
                if set(word_tokens).issubset(query_lemmas):
                    continue
            
            filtered.append(candidate)
        
        return filtered
    
    def get_top_tfidf_word(self, query: str, texts: List[str], 
                           top_n_candidates: int = 50,
                           local_weight: float = 0.6,
                           specificity_weight: float = 0.4) -> Tuple[str, List[Dict]]:
        """
        Get the best expansion word using local TF-IDF + category specificity.
        
        Args:
            query: Original query
            texts: Retrieved document texts
            top_n_candidates: Number of TF-IDF candidates to consider
            local_weight: Weight for local TF-IDF score (default: 0.6)
            specificity_weight: Weight for category specificity (default: 0.4)
            
        Returns:
            Tuple of (best_word, all_candidates_with_scores)
        """
        if not texts:
            return "", []
        
        # Lemmatize local texts (calculate_tfidf also lemmatizes internally)
        print(f"  Lemmatizing {len(texts)} local documents for specificity calculation...")
        lemmatized_local_texts = [lemmatize_text(text) for text in texts]
        
        # Calculate TF-IDF over retrieved documents (per-query, not global)
        tfidf_result = calculate_tfidf(texts, top_n=top_n_candidates, max_features=1000)
        candidates = tfidf_result['top_words']
        
        if not candidates:
            return "", []
        
        # Filter out query-related words
        filtered_candidates = self.filter_candidates(candidates, query)
        
        if not filtered_candidates:
            print("  No valid candidates after filtering")
            return "", []
        
        print(f"  Evaluating {len(filtered_candidates)} candidates...")
        
        # Calculate category specificity using document frequency
        # IMPORTANT: Use lemmatized texts since candidates come from lemmatized TF-IDF!
        specificity_scores = []
        has_global = bool(self.global_corpus)
        
        for candidate in filtered_candidates:
            spec_score = self.calculate_category_specificity(
                candidate['word'],
                lemmatized_local_texts  # Use LEMMATIZED texts for consistency
            )
            specificity_scores.append(spec_score)
        
        # Debug top 3 specificity scores
        if has_global and len(filtered_candidates) >= 3:
            print(f"  Top 3 specificity scores (doc frequency based):")
            for i in range(min(3, len(filtered_candidates))):
                cand = filtered_candidates[i]
                word_lower = cand['word'].lower()
                # Use lemmatized texts for accurate document frequency
                local_df = sum(1 for text in lemmatized_local_texts if word_lower in text.lower())
                global_df = sum(1 for text in self.global_corpus if word_lower in text.lower())
                spec = specificity_scores[i]
                print(f"    '{cand['word']}': local_df={local_df}/{len(lemmatized_local_texts)}, global_df={global_df}/{len(self.global_corpus)}, spec={spec:.4f}")
        
        # Extract scores
        local_tfidf = np.array([c['tfidf_score'] for c in filtered_candidates])
        spec_scores = np.array(specificity_scores)
        
        # Normalize
        local_norm = self.normalize_scores(local_tfidf)
        spec_norm = self.normalize_scores(spec_scores)
        
        
        # Combined score
        combined_scores = (local_weight * local_norm + 
                          specificity_weight * spec_norm)
        
        # Build results
        results = []
        for i, (candidate, local_n, spec_n, combined) in enumerate(
            zip(filtered_candidates, local_norm, spec_norm, combined_scores)
        ):
            global_score = self.global_vocab_scores.get(candidate['word'].lower(), 0.0) if has_global else None
            
            results.append({
                'word': candidate['word'],
                'local_tfidf': float(candidate['tfidf_score']),
                'local_tfidf_normalized': float(local_n),
                'global_tfidf': float(global_score) if global_score is not None else None,
                'category_specificity': float(spec_scores[i]),
                'category_specificity_normalized': float(spec_n),
                'combined_score': float(combined),
                'rank': i + 1
            })
        
        # Sort by combined score
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Log top result
        if results:
            top = results[0]
            print(f"  ✓ Top: '{top['word']}' (combined: {top['combined_score']:.3f})")
            if has_global:
                print(f"    Local={top['local_tfidf']:.4f}, Global={top.get('global_tfidf', 0.0):.4f}, "
                      f"Spec={top['category_specificity']:.4f}")
        
        best_word = results[0]['word'] if results else ""
        return best_word, results


def rerank_with_cross_encoder(query: str, texts: List[str], 
                              initial_scores: List[float],
                              cross_encoder: Optional[CrossEncoder] = None,
                              model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
                              batch_size: int = 32) -> List[float]:
    """
    Rerank results using a cross-encoder for improved ranking quality.
    Uses batching for efficiency.
    
    Args:
        query: Search query
        texts: List of document texts (limit to top-K for efficiency)
        initial_scores: Initial similarity scores (e.g., cosine similarity)
        cross_encoder: Pre-loaded CrossEncoder model (optional)
        model_name: Cross-encoder model to use (if cross_encoder not provided)
        batch_size: Batch size for CE prediction
        
    Returns:
        List of reranked scores
    """
    if not texts:
        return []
    
    # Limit number of documents for efficiency (top-100 max recommended)
    if len(texts) > 100:
        print(f"WARNING: Reranking {len(texts)} docs is expensive. Consider using top-100 only.")
    
    print(f"Reranking {len(texts)} results with cross-encoder (batch_size={batch_size})...")
    
    # Use provided cross-encoder or load new one
    if cross_encoder is None:
        cross_encoder = CrossEncoder(model_name)
    
    # Create query-document pairs
    pairs = [[query, text] for text in texts]
    
    # Get cross-encoder scores with batching
    ce_scores = cross_encoder.predict(pairs, show_progress_bar=False, batch_size=batch_size)
    ce_scores = np.array(ce_scores)
    
    # Normalize scores to [0, 1] using min-max
    if ce_scores.max() > ce_scores.min():
        ce_scores_norm = (ce_scores - ce_scores.min()) / (ce_scores.max() - ce_scores.min())
    else:
        ce_scores_norm = ce_scores
    
    # Normalize initial scores
    initial_scores = np.array(initial_scores)
    if initial_scores.max() > initial_scores.min():
        initial_scores_norm = (initial_scores - initial_scores.min()) / (initial_scores.max() - initial_scores.min())
    else:
        initial_scores_norm = initial_scores
    
    # Combine with initial scores (80% cross-encoder, 20% initial)
    combined_scores = 0.8 * ce_scores_norm + 0.2 * initial_scores_norm
    
    return combined_scores.tolist()





def add_tfidf_to_results(input_path: str, output_path: str, top_n: int = 20):
    """
    Add TF-IDF analysis to search results JSON.
    Supports both JSON and JSONL formats.
    
    Args:
        input_path: Path to input JSON or JSONL file
        output_path: Path to save enhanced JSON
        top_n: Number of top TF-IDF words to include
    """
    # Load search results (handles both JSON and JSONL)
    results = load_search_results(input_path)
    
    # Extract review texts
    review_texts = [review['text'] for review in results['reviews']]
    
    if not review_texts:
        print("No reviews found in search results!")
        results['tfidf_analysis'] = {
            'top_words': [],
            'total_unique_terms': 0,
            'total_documents': 0
        }
    else:
        print(f"Calculating TF-IDF for {len(review_texts)} reviews...")
        
        # Calculate TF-IDF per-query (not global)
        tfidf_results = calculate_tfidf(review_texts, top_n=top_n)
        
        # Add to results
        results['tfidf_analysis'] = tfidf_results
        
        print(f"\nTop {min(10, top_n)} TF-IDF terms:")
        for i, word_data in enumerate(tfidf_results['top_words'][:10], 1):
            print(f"{i}. {word_data['word']}: {word_data['tfidf_score']:.4f}")
    
    # Save enhanced results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved enhanced results to {output_path}")
    
    return results


def main():
    """Main function to process search results."""
    
    # Process search results
    enhanced_results = add_tfidf_to_results(
        input_path='search_results.json',
        output_path='search_results_with_tfidf.json',
        top_n=20
    )
    
    # Print summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Query: {enhanced_results.get('query', 'unknown')}")
    print(f"Total reviews analyzed: {enhanced_results['total_above_threshold']}")
    print(f"Unique terms found: {enhanced_results['tfidf_analysis']['total_unique_terms']}")
    print(f"\nTop 10 most distinctive terms:")
    for i, word_data in enumerate(enhanced_results['tfidf_analysis']['top_words'][:10], 1):
        print(f"  {i}. {word_data['word']}: {word_data['tfidf_score']:.4f}")


if __name__ == "__main__":
    main()
