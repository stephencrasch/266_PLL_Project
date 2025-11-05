"""
Test to verify cosine similarity calculation between query and reviews.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("="*60)
print("Cosine Similarity Verification Test")
print("="*60)

# Load the same model used in your pipeline
model_name = 'all-MiniLM-L6-v2'
print(f"\nLoading model: {model_name}")
model = SentenceTransformer(model_name)
print("✓ Model loaded")

# Test cases
test_cases = [
    {
        "query": "car wash",
        "texts": [
            "came to brasserie again for lunch the other day, and had another great meal.",
            "Great car wash! My car looks brand new after the detailing service.",
            "They did an amazing job washing and waxing my vehicle.",
            "The food here is delicious and the service is excellent."
        ]
    },
    {
        "query": "plumbing contractor",
        "texts": [
            "Fixed my water leak quickly and professionally.",
            "Great pizza and pasta, highly recommend this Italian restaurant.",
            "The plumber replaced all our pipes and did excellent work.",
            "Love the ambiance and the food was fantastic."
        ]
    },
    {
        "query": "mexican restaurant",
        "texts": [
            "Best tacos in town! Authentic Mexican cuisine.",
            "They fixed my leaking faucet in under an hour.",
            "The car detailing service was top notch.",
            "Delicious enchiladas and great margaritas!"
        ]
    }
]

print("\n" + "="*60)
print("Running Similarity Tests")
print("="*60)

for test in test_cases:
    query = test["query"]
    texts = test["texts"]
    
    print(f"\n\nQuery: '{query}'")
    print("-" * 60)
    
    # Encode query
    query_embedding = model.encode([query], normalize_embeddings=True)
    
    # Encode texts
    text_embeddings = model.encode(texts, normalize_embeddings=True)
    
    # Calculate cosine similarities
    similarities = cosine_similarity(query_embedding, text_embeddings)[0]
    
    # Sort by similarity
    sorted_indices = np.argsort(similarities)[::-1]
    
    print("\nResults (sorted by similarity):")
    for rank, idx in enumerate(sorted_indices, 1):
        sim = similarities[idx]
        text = texts[idx]
        
        # Determine if this is a good match
        is_relevant = "✓" if rank <= 2 and sim > 0.5 else "✗"
        
        print(f"\n{rank}. Similarity: {sim:.4f} {is_relevant}")
        print(f"   Text: {text[:80]}...")
    
    # Check if top match makes sense
    top_sim = similarities[sorted_indices[0]]
    top_text = texts[sorted_indices[0]]
    
    print(f"\n{'='*60}")
    if top_sim > 0.7:
        print(f"⚠ WARNING: Very high similarity ({top_sim:.4f}) detected")
        print(f"Top text: {top_text[:100]}")
        
        # Check if it's actually relevant
        query_words = set(query.lower().split())
        text_words = set(top_text.lower().split())
        overlap = query_words & text_words
        
        if len(overlap) == 0:
            print("❌ PROBLEM: Top match has NO word overlap with query!")
        else:
            print(f"✓ Word overlap: {overlap}")

print("\n" + "="*60)
print("Test Complete")
print("="*60)
print("\nExpected behavior:")
print("  • 'car wash' should match car/detailing reviews highest")
print("  • 'plumbing contractor' should match plumber/leak reviews highest")
print("  • 'mexican restaurant' should match taco/enchilada reviews highest")
print("  • Restaurant reviews should NOT score 0.72 with 'car wash'")
