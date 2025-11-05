import pandas as pd
from embeddings import ReviewEmbeddingsCreator
import json
import importlib.util
import sys
import torch

# Import tf-idf_word module (handle hyphen in filename)
spec = importlib.util.spec_from_file_location("tf_idf_word", "tf-idf_word.py")
tf_idf_word = importlib.util.module_from_spec(spec)
sys.modules["tf_idf_word"] = tf_idf_word
spec.loader.exec_module(tf_idf_word)

# Import the class and functions
QueryExpander = tf_idf_word.QueryExpander
calculate_tfidf = tf_idf_word.calculate_tfidf


def main():
    """Run the complete review analysis pipeline."""
    
    # Define categories/queries to search
    
    categories = [
        'Cocktail Bar bartender friendly atmosphere',
        'Dive Bar fun pool table', 
        'Pubs friendly local beer',
        'Sports Bar game watch beer',
        'Wine Bar selection glass bottle'
    ]
    
    # Configuration
    similarity_threshold = .5
    k_results = 100
    top_n_tfidf = 20
    
    print("="*60)
    print("STEP 1: Loading Reviews and Creating Embeddings")
    print("="*60)
    
    # Load the prepared reviews
    reviews_df = pd.read_csv('reviews_for_embeddings.csv')
    print(f"Loaded {len(reviews_df)} reviews")
    
    run_embeddings = 0  # Set to 1 to rebuild embeddings
    
    if run_embeddings == 1:
        # Initialize embeddings creator
        creator = ReviewEmbeddingsCreator(model_name='all-MiniLM-L6-v2')

        # Store metadata
        creator.review_metadata = reviews_df[['review_id', 'business_id', 'stars']].to_dict('records')
    
        # Create embeddings
        embeddings = creator.create_embeddings(reviews_df, batch_size=32)
        
        # Build FAISS index
        creator.build_faiss_index(embeddings, index_type='flat')
        
        # Save index and metadata
        creator.save(
            index_path='review_embeddings.faiss',
            metadata_path='review_metadata.pkl'
        )
        
    else:
        # Load existing index and metadata
        creator = ReviewEmbeddingsCreator(model_name='all-MiniLM-L6-v2')
        creator.load(
            index_path='review_embeddings.faiss',
            metadata_path='review_metadata.pkl'
        )
    
    # Initialize query expander with global corpus for category specificity
    print("\nInitializing query expander with global corpus...")
    all_review_texts = reviews_df['text'].tolist()
    
    # Let QueryExpander auto-detect the best device (will use MPS on M3 Mac)
    expander = QueryExpander(
        cross_encoder_model='cross-encoder/ms-marco-MiniLM-L-6-v2',
        global_corpus=all_review_texts,  # For category specificity
        device=None  # Auto-detect: prefers MPS > CUDA > CPU
    )
    
    print("\n" + "="*60)
    print("STEP 2: Searching for Categories and Running TF-IDF")
    print("="*60)
    
    # Store all results in a single structure
    all_results = {
        'similarity_threshold': similarity_threshold,
        'k_results': k_results,
        'total_reviews_corpus': len(reviews_df),
        'categories': []
    }
    
    for i, category in enumerate(categories, 1):
        print(f"\n[{i}/{len(categories)}] Processing: {category}")
        print("-" * 60)
        
        # Get similar reviews
        similar_reviews = creator.get_similar_reviews(category, reviews_df, k=k_results)
        
        # Filter by similarity threshold
        filtered_reviews = similar_reviews[similar_reviews['cosine_similarity'] >= similarity_threshold]
        
        print(f"Found {len(filtered_reviews)} reviews with similarity >= {similarity_threshold}")
        
        # Build category result
        category_result = {
            'category': category,
            'total_retrieved': len(similar_reviews),
            'total_above_threshold': len(filtered_reviews),
            'reviews': []
        }
        
        # Add reviews
        for idx, row in filtered_reviews.iterrows():
            review_data = {
                'review_id': row['review_id'],
                'business_id': row['business_id'],
                'cosine_similarity': float(row['cosine_similarity']),
                'text': row['text'],
                'label': row['label'] if 'label' in row else None
            }
            category_result['reviews'].append(review_data)
        
        # Calculate TF-IDF and get expansion word if we have results
        if len(filtered_reviews) > 0:
            print(f"Calculating TF-IDF for {category}...")
            review_texts = [review['text'] for review in category_result['reviews']]
            
            tfidf_analysis = calculate_tfidf(review_texts, top_n=top_n_tfidf)
            category_result['tfidf_analysis'] = tfidf_analysis
            
            # Show top 5 TF-IDF terms
            if tfidf_analysis['top_words']:
                print(f"\nTop 5 TF-IDF terms for '{category}':")
                for j, word_data in enumerate(tfidf_analysis['top_words'][:5], 1):
                    print(f"  {j}. {word_data['word']}: {word_data['tfidf_score']:.4f}")
            
            # Get best expansion word
            print(f"\nFinding best expansion word...")
            expansion_word, all_candidates = expander.get_top_tfidf_word(
                query=category,
                texts=review_texts,
                top_n_candidates=50,
                num_docs_for_scoring=20,
                local_weight=0.3,  # Local TF-IDF
                ce_weight=0.5,  # Cross-encoder
                specificity_weight=0.2  # Category specificity
            )
            
            # Add expansion info to result
            category_result['query_expansion'] = {
                'suggested_word': expansion_word,
                'expanded_query': f"{category} {expansion_word}" if expansion_word else category,
                'top_candidates': all_candidates[:10]  # Save top 10 for analysis
            }
            
        
        else:
            category_result['tfidf_analysis'] = {
                'top_words': [],
                'total_unique_terms': 0,
                'total_documents': 0
            }
            category_result['query_expansion'] = {
                'suggested_word': "",
                'expanded_query': category,
                'top_candidates': []
            }
        
        # Add to all results
        all_results['categories'].append(category_result)
    
    # Save all results to a single JSON file
    combined_output = 'all_category_results_with_tfidf.json'
    with open(combined_output, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved all results to {combined_output}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nProcessed {len(categories)} categories")
    print(f"Similarity threshold: {similarity_threshold}")
    print(f"Results per category (max): {k_results}")
    print(f"Total reviews in corpus: {len(reviews_df)}")
    print(f"\nResults by category:")
    for cat_result in all_results['categories']:
        print(f"  • {cat_result['category']}: {cat_result['total_above_threshold']} reviews")
    
    print(f"\nOutput file:")
    print(f"  • {combined_output}")
    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()
