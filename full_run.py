"""
Full pseudo-label training pipeline (Round 0 style)
1. Retrieve similar reviews for each category using FAISS.
2. Compute TF-IDF + expansion word.
3. Save results to JSON.
4. Fine-tune ModernBERT on pseudo-labeled data.
"""

import os
import re
import json
import pandas as pd
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding, set_seed,
    EarlyStoppingCallback
)
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import importlib.util
import sys
from embeddings import ReviewEmbeddingsCreator

# ---- import tf-idf_word (filename contains hyphen) ----
spec = importlib.util.spec_from_file_location("tf_idf_word", "tf-idf_word.py")
tf_idf_word = importlib.util.module_from_spec(spec)
sys.modules["tf_idf_word"] = tf_idf_word
spec.loader.exec_module(tf_idf_word)
QueryExpander = tf_idf_word.QueryExpander
calculate_tfidf = tf_idf_word.calculate_tfidf




# =========================
# STEP 1: CONFIG
# =========================
set_seed(42)

# SETUP CONFIGURATION
create_embeddings_and_split = False  # Set to True to create initial train/holdout split and FAISS index
# Only set to True the FIRST TIME you run this script, then set to False for all subsequent runs

# ROUND CONFIGURATION - CHANGE THIS FOR EACH ROUND
current_round = 1  # Change to 1, 2, 3, etc. for subsequent rounds
prev_model_path = None  # For round 0
# prev_model_path = "./modernbert_r0"  # Uncomment and set for round 1+
use_model_filtering = True  # Set to True for round 1+ to filter docs by model predictions

# For Round 1+: Load expanded queries from previous round
prev_round_json = f"all_category_results_r{current_round - 1}.json" if current_round > 0 else None

categories = ['Cocktail Bars',
        'Dive Bars', 
        'Pubs',
        'Sports Bars',
        'Wine Bars']

# Retrieval config
k_biencoder = 600  # Bi-encoder initial retrieval
k_crossencoder = 300  # Cross-encoder re-ranking (top k to keep)
ce_threshold = 0.0  # Cross-encoder score threshold (CE scores typically range -10 to 10)

# Beam search config
num_beams = 3  # Number of expansion beams per category (1 = no beam search)
beam_strategy = "union"  # "union" (merge all beams), "top_beam" (use best beam only)

# Training data config
samples_per_class = 100  # Set to enforce per-class cap (e.g., 100), or None to use all retrieved

top_n_tfidf = 100  # Number of top TF-IDF words to consider for expansion
round_name = f"r{current_round}"
json_output = f"all_category_results_{round_name}.json"

embedding_model = 'all-MiniLM-L6-v2'
cross_encoder_model = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
reviews_csv = 'reviews_for_embeddings.csv'

# modernbert_model = "answerdotai/ModernBERT-base"
modernbert_model = "distilbert-base-uncased"


output_dir = f"./modernbert_{round_name}"
epochs = 3  

# Device setup for M3 Mac
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")
print(f"Round: {current_round}")
if prev_model_path:
    print(f"Will load previous model from: {prev_model_path}")
    print(f"Model filtering for TF-IDF: {use_model_filtering}")


# =========================
# STEP 2: LOAD/CREATE EMBEDDINGS + FAISS INDEX
# =========================
print("=" * 60)
print("STEP 2: Load/Create embeddings and FAISS index")
print("=" * 60)

creator = ReviewEmbeddingsCreator(model_name=embedding_model)

if create_embeddings_and_split:
    # First time setup: create train/dev/test split and FAISS index
    print("\n⚠️  CREATE MODE: Building train/dev/test split and FAISS index...")
    print("This should only be run ONCE at the beginning!")
    
    train_df, dev_df, test_df = creator.setup_from_reviews_csv(
        reviews_csv=reviews_csv,
        dev_size=0.15,  # 15% for dev (early stopping)
        test_size=0.20,  # 20% for test (final evaluation)
        stratify_column='label',
        random_state=42,
        batch_size=32,
        index_type='flat',
        index_path='review_embeddings_train.faiss',
        metadata_path='review_metadata.pkl',
        train_path='reviews_train.csv',
        dev_path='reviews_dev.csv',
        test_path='reviews_test.csv'
    )
    
    reviews_df = train_df
    
else:
    # Normal mode: load existing index and splits
    print("\n📂 LOAD MODE: Loading existing index and data splits...")
    
    creator.load(
        index_path="review_embeddings_train.faiss",
        metadata_path="review_metadata.pkl",
        train_path="reviews_train.csv",
        dev_path="reviews_dev.csv",
        test_path="reviews_test.csv"
    )
    
    # Use training data only
    reviews_df = creator.train_df
    print(f"\nUsing {len(reviews_df):,} training reviews for retrieval")
    print(f"Dev set has {len(creator.dev_df):,} reviews (for early stopping)")
    print(f"Test set has {len(creator.test_df):,} reviews (for FINAL evaluation ONLY)")


# =========================
# STEP 2b: LOAD PREVIOUS MODEL (FOR ROUND 1+)
# =========================
prev_model = None
prev_tokenizer = None

if use_model_filtering and prev_model_path:
    print("\n" + "=" * 60)
    print("STEP 2b: Load previous round model for filtering")
    print("=" * 60)
    
    prev_tokenizer = AutoTokenizer.from_pretrained(prev_model_path)
    prev_model = AutoModelForSequenceClassification.from_pretrained(
        prev_model_path
    ).to(device)
    prev_model.eval()
    
    print(f"Loaded model from {prev_model_path}")
    print(f"Model has {prev_model.config.num_labels} labels: {list(prev_model.config.id2label.values())}")

# =========================
# STEP 3: RETRIEVAL + RE-RANKING + TF-IDF + EXPANSION
# =========================
print("\n" + "=" * 60)
print("STEP 3: Retrieve and re-rank similar reviews per category")
print("=" * 60)

# Load expanded queries from previous round (if Round 1+)
expanded_queries = {}  # Format: {category: [beam1_query, beam2_query, beam3_query]}
if current_round > 0 and prev_round_json and os.path.exists(prev_round_json):
    print(f"\n📖 Loading expanded queries from {prev_round_json}")
    with open(prev_round_json, 'r') as f:
        prev_results = json.load(f)
    
    for cat_result in prev_results.get("categories", []):
        category = cat_result["category"]
        # Load beam queries if they exist, otherwise use single query
        beams = cat_result.get("beams", [])
        if beams:
            # Round 1+: Load EXPANDED queries from previous round (not base queries!)
            beam_queries = [beam.get("expanded_query", category) for beam in beams]
            expanded_queries[category] = beam_queries[:num_beams]  # Take up to num_beams
            print(f"  {category} → {len(beam_queries)} beams:")
            for idx, q in enumerate(beam_queries[:num_beams], 1):
                print(f"    Beam {idx}: '{q}'")
        else:
            # Fallback: single query from old format
            expansion = cat_result.get("query_expansion", {})
            single_query = expansion.get("expanded_query", category)
            expanded_queries[category] = [single_query] * num_beams  # Replicate for all beams
            print(f"  {category} → '{single_query}' (replicated to {num_beams} beams)")
else:
    print("\n📝 Round 0: Using original category names as queries")
    expanded_queries = {cat: [cat] * num_beams for cat in categories}  # Start all beams with base category

# Initialize cross-encoder for re-ranking on MPS
from sentence_transformers import CrossEncoder
cross_encoder = CrossEncoder(cross_encoder_model, device=device)
print(f"Loaded cross-encoder: {cross_encoder_model} on {device}")

# Initialize QueryExpander with caching for global IDF
all_review_texts = reviews_df["text"].astype(str).tolist()
expander_cache_path = "query_expander_cache.pkl"

# Try to load cached expander
if os.path.exists(expander_cache_path) and not create_embeddings_and_split:
    print(f"📦 Loading cached QueryExpander from {expander_cache_path}...")
    import pickle
    try:
        with open(expander_cache_path, 'rb') as f:
            expander = pickle.load(f)
        print(f"✅ Loaded cached global IDF ({len(expander.global_vocab_scores)} terms)")
    except Exception as e:
        print(f"⚠️  Failed to load cache ({e}), recomputing...")
        print(f"Computing global IDF from {len(all_review_texts):,} training reviews...")
        expander = QueryExpander(global_corpus=all_review_texts)
        # Save for future rounds
        with open(expander_cache_path, 'wb') as f:
            pickle.dump(expander, f)
        print(f"✅ Cached QueryExpander to {expander_cache_path}")
else:
    print(f"Computing global IDF from {len(all_review_texts):,} training reviews...")
    expander = QueryExpander(global_corpus=all_review_texts)
    # Save for future rounds
    import pickle
    with open(expander_cache_path, 'wb') as f:
        pickle.dump(expander, f)
    print(f"✅ Cached QueryExpander to {expander_cache_path}")

all_results = {
    "round": round_name,
    "k_biencoder": k_biencoder,
    "k_crossencoder": k_crossencoder,
    "ce_threshold": ce_threshold,
    "total_reviews_corpus": len(reviews_df),
    "used_expanded_queries": current_round > 0,
    "categories": []
}

for i, category in enumerate(categories, 1):
    # Get beam queries for this category
    beam_queries = expanded_queries.get(category, [category] * num_beams)
    
    print(f"\n[{i}/{len(categories)}] {category}")
    print(f"  Running {len(beam_queries)} beams:")
    
    # Store results for each beam
    all_beams = []
    
    # Step 3: Retrieve and re-rank for EACH beam
    for beam_idx, query in enumerate(beam_queries, 1):
        print(f"\n  Beam {beam_idx}/{len(beam_queries)}: '{query}'")
        
        # Step 3a: Bi-encoder retrieval with expanded query
        similar_reviews = creator.get_similar_reviews(query, reviews_df, k=k_biencoder)
        print(f"    Bi-encoder retrieved {len(similar_reviews)} reviews")
        
        # Step 3b: Cross-encoder re-ranking
        if len(similar_reviews) > 0:
            # Prepare query-document pairs for cross-encoder
            query_doc_pairs = [
                (query, str(row["text"]))
                for _, row in similar_reviews.iterrows()
            ]
            
            # Get cross-encoder scores (batched for efficiency on MPS)
            ce_scores = cross_encoder.predict(query_doc_pairs, batch_size=32, show_progress_bar=False)
            
            # Add cross-encoder scores and beam info to dataframe
            similar_reviews["ce_score"] = ce_scores
            similar_reviews["beam_id"] = beam_idx
            similar_reviews["beam_query"] = query
            
            # Sort by cross-encoder score and take top k_crossencoder
            similar_reviews = similar_reviews.nlargest(k_crossencoder, "ce_score").reset_index(drop=True)
            print(f"    After re-ranking: kept top {len(similar_reviews)} reviews")
            
            # Apply threshold (using CE scores)
            filtered = similar_reviews[similar_reviews["ce_score"] >= ce_threshold]
            print(f"    After threshold {ce_threshold}: {len(filtered)} reviews remain")
        else:
            filtered = similar_reviews
        
        all_beams.append({
            'beam_id': beam_idx,
            'query': query,
            'reviews': filtered
        })
    
    # Step 3c: Merge beams according to strategy
    print(f"\n  Merging {len(all_beams)} beams using '{beam_strategy}' strategy...")
    
    if beam_strategy == "union":
        # Union: Combine all beams, keep best CE score for duplicates
        all_beam_reviews = []
        for beam in all_beams:
            all_beam_reviews.append(beam['reviews'])
        
        if all_beam_reviews:
            merged = pd.concat(all_beam_reviews, ignore_index=True)
            # Deduplicate by review_id, keeping highest CE score
            merged = (merged
                     .sort_values('ce_score', ascending=False)
                     .drop_duplicates(subset=['review_id'], keep='first')
                     .reset_index(drop=True))
            print(f"  After union merge: {len(merged)} unique reviews")
        else:
            merged = pd.DataFrame()
            
    elif beam_strategy == "top_beam":
        # Top beam: Use only the beam with highest average CE score
        beam_avg_scores = []
        for beam in all_beams:
            if len(beam['reviews']) > 0:
                avg_score = beam['reviews']['ce_score'].mean()
                beam_avg_scores.append((beam['beam_id'], avg_score, beam['reviews']))
        
        if beam_avg_scores:
            beam_avg_scores.sort(key=lambda x: x[1], reverse=True)
            best_beam_id, best_avg, merged = beam_avg_scores[0]
            print(f"  Selected beam {best_beam_id} (avg CE score: {best_avg:.4f})")
        else:
            merged = pd.DataFrame()
    else:
        # Default to union if strategy unknown
        merged = pd.concat([b['reviews'] for b in all_beams], ignore_index=True)
        merged = merged.drop_duplicates(subset=['review_id'], keep='first').reset_index(drop=True)
    
    filtered = merged

    cat_result = {
        "category": category,
        "num_beams": len(beam_queries),
        "beam_strategy": beam_strategy,
        "total_biencoder_retrieved": k_biencoder,
        "total_after_merging": len(filtered),
        "beams": [],  # Will store beam info
        "reviews": []
    }

    for _, row in filtered.iterrows():
        cat_result["reviews"].append({
            "review_id": row["review_id"],
            "business_id": row["business_id"],
            "cosine_similarity": float(row["cosine_similarity"]),
            "ce_score": float(row["ce_score"]),  # Added: cross-encoder score
            "text": str(row["text"])
        })

    if len(filtered) > 0:
        texts = [r["text"] for r in cat_result["reviews"]]
        
        # Compute TF-IDF on merged results
        tfidf = calculate_tfidf(texts, top_n=top_n_tfidf)
        cat_result["tfidf_analysis"] = tfidf
        
        # Generate expansion word for EACH beam
        print(f"\n  Generating {num_beams} expansion words (one per beam)...")
        
        # Optional: Filter texts using model predictions for cleaner TF-IDF (Round 1+)
        texts_for_tfidf = texts
        if use_model_filtering and prev_model_path and current_round > 0:
            print(f"\n  🔍 Filtering docs with R{current_round-1} model predictions for category: {category}")
            
            # Try to load cached predictions from previous round
            prev_predictions_path = f"predictions_r{current_round - 1}.csv"
            if os.path.exists(prev_predictions_path):
                pred_df = pd.read_csv(prev_predictions_path)
                
                # Create lookup: review_id -> (predicted_label, confidence)
                pred_lookup = dict(zip(
                    pred_df['review_id'], 
                    zip(pred_df['predicted_label'], pred_df['confidence'])
                ))
                
                # Filter to only documents predicted as this category with confidence threshold
                conf_threshold = 0.6  # Tune this (0.4-0.6 range typically works)
                filtered_indices = []
                
                for idx, r in enumerate(cat_result["reviews"]):
                    review_id = r['review_id']
                    if review_id in pred_lookup:
                        predicted_label, confidence = pred_lookup[review_id]
                        if predicted_label == category and confidence >= conf_threshold:
                            filtered_indices.append(idx)
                
                texts_for_tfidf = [texts[i] for i in filtered_indices]
                
                # Fallback: if too few docs after filtering, use all (prevents collapse)
                min_docs_for_tfidf = 50
                if len(texts_for_tfidf) < min_docs_for_tfidf:
                    print(f"  ⚠️  Only {len(texts_for_tfidf)} docs after filtering (< {min_docs_for_tfidf}), using all {len(texts)} docs")
                    texts_for_tfidf = texts
                else:
                    print(f"  ✅ Filtered: {len(texts_for_tfidf)}/{len(texts)} docs predicted as '{category}' (conf ≥ {conf_threshold})")
                
            else:
                # Fall back to on-the-fly computation if cached predictions not found
                print(f"  ⚠️  {prev_predictions_path} not found, computing predictions on-the-fly...")
                
                if prev_model is None:
                    # Load model if not already loaded
                    prev_tokenizer = AutoTokenizer.from_pretrained(prev_model_path)
                    prev_model = AutoModelForSequenceClassification.from_pretrained(
                        prev_model_path
                    ).to(device)
                    prev_model.eval()
                
                # Get model predictions for all retrieved texts
                inputs = prev_tokenizer(texts, truncation=True, max_length=512, padding=True, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = prev_model(**inputs)
                    logits = outputs.logits
                    preds = logits.argmax(dim=-1).cpu().numpy()
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
                    confidences = probs.max(axis=-1)
                
                # Get the label ID for current category
                category_id = prev_model.config.label2id.get(category)
                
                if category_id is not None:
                    # Filter to only documents predicted as this category with confidence
                    conf_threshold = 0.5
                    filtered_indices = [
                        i for i, (pred, conf) in enumerate(zip(preds, confidences))
                        if pred == category_id and conf >= conf_threshold
                    ]
                    texts_for_tfidf = [texts[i] for i in filtered_indices]
                    
                    # Fallback: use all if too few
                    min_docs_for_tfidf = 50
                    if len(texts_for_tfidf) < min_docs_for_tfidf:
                        print(f"  ⚠️  Only {len(texts_for_tfidf)} docs after filtering, using all {len(texts)} docs")
                        texts_for_tfidf = texts
                    else:
                        print(f"  ✅ Filtered: {len(texts_for_tfidf)}/{len(texts)} docs predicted as '{category}' (conf ≥ {conf_threshold})")
                else:
                    print(f"  ⚠️  '{category}' not found in model labels, using all docs")
                    texts_for_tfidf = texts
        
        # Compute TF-IDF on (possibly filtered) texts
        tfidf = calculate_tfidf(texts_for_tfidf, top_n=top_n_tfidf)
        cat_result["tfidf_analysis"] = tfidf

        # Generate expansion word for EACH beam
        print(f"  Generating {num_beams} expansion words (one per beam)...")
        
        beam_expansions = []
        used_words = set()  # Track words already used across beams to prevent duplicates
        
        # Check if all beams have the same query (Round 0: need to generate multiple words from same query)
        all_queries_same = len(set(beam_queries)) == 1
        
        if all_queries_same:
            # Round 0: All beams start with same query, get multiple expansion words
            print(f"    All beams have same query '{beam_queries[0]}' - generating {num_beams} different expansions...")
            _, all_candidates = expander.get_top_tfidf_word(
                query=beam_queries[0],
                texts=texts_for_tfidf,
                top_n_candidates=50,  # Get more candidates to ensure enough diversity
                local_weight=0.4,
                specificity_weight=0.6
            )
            
            # Distribute top candidates to beams
            query_lower = beam_queries[0].lower()
            for beam_idx, beam_query in enumerate(beam_queries, 1):
                # Find next unused candidate not in query
                selected_word = None
                for candidate in all_candidates:
                    candidate_word = candidate.get('word', '')
                    candidate_lower = candidate_word.lower()
                    
                    if (candidate_word and 
                        candidate_lower not in query_lower and
                        candidate_lower not in used_words):
                        selected_word = candidate_word
                        used_words.add(candidate_lower)
                        break
                
                if selected_word:
                    new_query = f"{beam_query} {selected_word}"
                    print(f"      Beam {beam_idx}: '{beam_query}' → '{new_query}' (added: '{selected_word}')")
                else:
                    new_query = beam_query
                    print(f"      Beam {beam_idx}: No new expansion (keeping '{beam_query}')")
                
                beam_expansions.append({
                    "beam_id": beam_idx,
                    "query": beam_query,
                    "suggested_word": selected_word or "",
                    "expanded_query": new_query
                })
        else:
            # Round 1+: Each beam has different query, use beam-specific TF-IDF
            print(f"    Beams have different queries - using beam-specific expansion...")
            for beam_idx, beam_query in enumerate(beam_queries, 1):
                # Get beam-specific TF-IDF candidates using THIS beam's query
                print(f"      Beam {beam_idx}: Analyzing with query '{beam_query}'...")
                _, beam_candidates = expander.get_top_tfidf_word(
                    query=beam_query,  # ← Use THIS beam's query for context-specific expansion
                    texts=texts_for_tfidf,
                    top_n_candidates=30,  # Get enough candidates to avoid collisions
                    local_weight=0.4,
                    specificity_weight=0.6
                )
                
                # Get query as lowercase string for substring matching (handles bigrams)
                query_lower = beam_query.lower()
                
                # Find first unused candidate not in this beam's query
                selected_word = None
                for candidate in beam_candidates:
                    candidate_word = candidate.get('word', '')
                    candidate_lower = candidate_word.lower()
                    
                    # Check if candidate is already in query (substring match for bigrams)
                    # OR if candidate was already used by another beam
                    if (candidate_word and 
                        candidate_lower not in query_lower and
                        candidate_lower not in used_words):
                        selected_word = candidate_word
                        used_words.add(candidate_lower)
                        break
                
                if selected_word:
                    new_query = f"{beam_query} {selected_word}"
                    print(f"        → '{new_query}' (added: '{selected_word}')")
                else:
                    new_query = beam_query
                    print(f"        → No new expansion (keeping '{beam_query}')")
                
                beam_expansions.append({
                    "beam_id": beam_idx,
                    "query": beam_query,
                    "suggested_word": selected_word or "",
                    "expanded_query": new_query
                })
        
        # Store beam expansions
        cat_result["beams"] = beam_expansions
        # Store top candidates from first beam for analysis (beam-specific candidates available in beam details)
        cat_result["used_model_filtering"] = use_model_filtering and prev_model is not None
        cat_result["filtered_doc_count"] = len(texts_for_tfidf) if use_model_filtering else len(texts)
    else:
        cat_result["tfidf_analysis"] = {"top_words": [], "total_unique_terms": 0}
        cat_result["query_expansion"] = {
            "suggested_word": "", "expanded_query": category, "top_candidates": []
        }

    all_results["categories"].append(cat_result)

# Save round results
with open(json_output, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)
print(f"\nSaved results to {json_output}")

# =========================
# STEP 4: PREPARE DATA FOR TRAINING (SMART DEDUPLICATION)
# =========================
print("\n" + "=" * 60)
print("STEP 4: Prepare pseudo-labeled training data")
print("=" * 60)

# Collect all retrieved reviews with their scores
rows = []
for cat in all_results["categories"]:
    label = cat["category"]
    for r in cat["reviews"]:
        txt = (r.get("text") or "").strip()
        if txt:
            rows.append({
                "review_id": r["review_id"],
                "text": txt,
                "label": label,
                "ce_score": r["ce_score"],  # Cross-encoder score for this category
                "cosine_similarity": r["cosine_similarity"]
            })

df_all = pd.DataFrame(rows)
print(f"\nTotal retrievals before deduplication: {len(df_all)}")
print(f"Retrievals per category:\n{df_all['label'].value_counts()}")

# Deduplicate: Keep the category with highest CE score for each review
print("\n🔧 Deduplicating by review_id (keeping highest CE score per review)...")
df_deduped = (df_all
              .sort_values('ce_score', ascending=False)  # Sort by score
              .drop_duplicates(subset=['review_id'], keep='first')  # Keep best category
              .reset_index(drop=True))

duplicates_removed = len(df_all) - len(df_deduped)
print(f"After deduplication: {len(df_deduped)} unique reviews")
if duplicates_removed > 0:
    print(f"  → Removed {duplicates_removed} duplicates ({duplicates_removed/len(df_all)*100:.1f}%)")
    print(f"  → These reviews were retrieved by multiple categories")
else:
    print(f"  → No duplicates found (all reviews category-specific)")
print(f"Reviews per category after dedup:\n{df_deduped['label'].value_counts()}")

# Optionally enforce per-class cap (to match WANDER's top-N per class)
if samples_per_class is not None:
    print(f"\n🔧 Enforcing per-class cap: {samples_per_class} samples per class...")
    
    df_balanced = []
    for label in categories:
        class_df = df_deduped[df_deduped['label'] == label]
        # Take top N by CE score for this class
        class_df_top = class_df.nlargest(min(samples_per_class, len(class_df)), 'ce_score')
        df_balanced.append(class_df_top)
    
    df = pd.concat(df_balanced, ignore_index=True)
    print(f"\nFinal training set: {len(df)} reviews")
    print(f"Reviews per category (capped):\n{df['label'].value_counts()}")
else:
    df = df_deduped
    print(f"\nFinal training set: {len(df)} reviews (no per-class cap)")
    print(f"Reviews per category:\n{df['label'].value_counts()}")

# Keep only text and label for training
df = df[['text', 'label']].reset_index(drop=True)

# =========================
# STEP 5: TRAIN MODERNBERT WITH LABEL SMOOTHING & CLEAN DEV SET
# =========================
print("\n" + "=" * 60)
print("STEP 5: Fine-tune ModernBERT on pseudo labels")
print("=" * 60)

labels_sorted = sorted(df["label"].unique())
label2id = {lbl: i for i, lbl in enumerate(labels_sorted)}
id2label = {i: lbl for lbl, i in label2id.items()}
df["labels"] = df["label"].map(label2id)

# Use ALL pseudo-labeled data for training (no validation split from noisy data)
train_df = df
print(f"Training on {len(train_df)} pseudo-labeled reviews")

# Load CLEAN dev set for validation/early stopping
print(f"\nLoading clean dev set for early stopping...")
dev_df = pd.read_csv('reviews_dev.csv')
dev_df = dev_df[dev_df['label'].isin(labels_sorted)].copy()
dev_df['labels'] = dev_df['label'].map(label2id)
print(f"Dev set (for early stopping): {len(dev_df):,} reviews with TRUE labels")
print(f"Dev label distribution:\n{dev_df['label'].value_counts()}")

# Validate we have dev data
if len(dev_df) == 0:
    raise ValueError("Dev set is empty after filtering! Check that reviews_dev.csv has matching labels.")

print(f"\n⚠️  IMPORTANT: Test set (reviews_test.csv) will ONLY be used for final evaluation")
print(f"              It is NOT loaded or used during training!")

tokenizer = AutoTokenizer.from_pretrained(modernbert_model)
collator = DataCollatorWithPadding(tokenizer=tokenizer)

def tok(batch):
    return tokenizer(batch["text"], truncation=True, max_length=512)

# Prepare datasets
train_ds = Dataset.from_pandas(train_df[["text", "labels"]], preserve_index=False).map(tok, batched=True, remove_columns=["text"])
val_ds = Dataset.from_pandas(dev_df[["text", "labels"]], preserve_index=False).map(tok, batched=True, remove_columns=["text"])
ds = DatasetDict({"train": train_ds, "validation": val_ds})

# Debug: Check dataset sizes
print(f"\n🔍 Dataset Debug Info:")
print(f"  train_df shape: {train_df.shape}")
print(f"  dev_df shape: {dev_df.shape}")
print(f"  train_ds length: {len(train_ds)}")
print(f"  val_ds length: {len(val_ds)}")
print(f"  val_ds columns: {val_ds.column_names}")
if len(val_ds) > 0:
    print(f"  val_ds first example keys: {val_ds[0].keys()}")
else:
    print(f"  ❌ ERROR: val_ds is EMPTY after tokenization!")
    print(f"  dev_df preview:\n{dev_df.head()}")

# Initialize model (from previous round or fresh)
if current_round > 0 and prev_model_path:
    # Round 1+: Continue training from previous round
    print(f"\n🔄 Round {current_round}: Initializing from previous round model")
    print(f"   Loading weights from: {prev_model_path}")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        prev_model_path,  # ← Load trained weights from previous round
        num_labels=len(labels_sorted),
        id2label=id2label,
        label2id=label2id,
    )
    
    # Enable label smoothing in model config
    model.config.problem_type = "single_label_classification"
    
    print(f"✅ Successfully loaded Round {current_round-1} model")
    print(f"   Model will be fine-tuned on new pseudo-labels from improved queries")
    
else:
    # Round 0: Start from pre-trained ModernBERT
    print(f"\n🆕 Round 0: Initializing fresh ModernBERT model")
    print(f"   Loading pre-trained weights from: {modernbert_model}")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        modernbert_model,
        num_labels=len(labels_sorted),
        id2label=id2label,
        label2id=label2id,
    )
    
    # Enable label smoothing in model config
    model.config.problem_type = "single_label_classification"
    
    print(f"✅ Fresh model initialized for Round 0")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }

os.makedirs(output_dir, exist_ok=True)

# Calculate steps per epoch for dynamic eval frequency
batch_size = 8
steps_per_epoch = len(train_df) // batch_size
eval_steps = max(20, steps_per_epoch // 2)  # Evaluate 2x per epoch, min 20 steps
print(f"\nTraining configuration:")
print(f"  Training samples: {len(train_df)}")
print(f"  Dev samples: {len(dev_df)}")
print(f"  Batch size: {batch_size}")
print(f"  Steps per epoch: ~{steps_per_epoch}")
print(f"  Evaluating every {eval_steps} steps")
print(f"  Dev batches per eval: {len(dev_df) // 16} (batch_size=16)")

# Ensure we have enough dev data for at least one batch
if len(dev_df) < 16:
    print(f"⚠️  Warning: Dev set ({len(dev_df)}) smaller than eval batch size (16). Metrics may be unreliable.")

args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="steps",
    eval_steps=eval_steps,  # Dynamic based on dataset size
    save_strategy="steps",
    save_steps=eval_steps,  # Save at same frequency
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",  # Stop based on F1 on CLEAN dev set
    greater_is_better=True,
    num_train_epochs=epochs,  # Default 3 epochs with early stopping
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=16,
    learning_rate=1e-5,  # Lower learning rate for stability
    weight_decay=0.05,  # Increased weight decay for regularization
    warmup_ratio=0.1,  # Increased warmup
    logging_steps=max(10, steps_per_epoch // 5),  # Log 5x per epoch
    save_total_limit=3,
    report_to="none",
    fp16=False,  # Don't use fp16 on MPS
    label_smoothing_factor=0.1,  # Built-in label smoothing (0.1 = 10% smoothing)
)

# Create trainer with early stopping
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],  # CLEAN holdout set for validation
    tokenizer=tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # Stop if no improvement for 3 evals
)

print("\n" + "=" * 60)
print("TRAINING WITH:")
print("  ✓ Label Smoothing (0.1) - prevents overconfidence on noisy pseudo-labels")
print("  ✓ Clean Dev Set - uses TRUE labels for early stopping (NOT test set!)")
print("  ✓ Early Stopping (patience=3) - stops when dev F1 stops improving")
print("  ✓ Increased Regularization - weight decay 0.05")
print("  ✓ Frequent Evaluation - every 100 steps for better stopping point")
print("  ✓ Test Set Reserved - reviews_test.csv will ONLY be used for final evaluation")
print("=" * 60)

trainer.train()

# Evaluate on holdout (final check)
print("\n" + "=" * 60)
print("FINAL EVALUATION ON CLEAN DEV SET")
print("=" * 60)
eval_results = trainer.evaluate()
print(f"Dev Accuracy: {eval_results.get('eval_accuracy', 0):.4f}")
print(f"Dev F1 (Macro): {eval_results.get('eval_f1_macro', 0):.4f}")

trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"\n✅ Training complete. Best model saved to {output_dir}")
print(f"✅ Model was selected based on BEST F1 on clean DEV data")

# =========================
# STEP 6: SAVE PREDICTIONS FOR NEXT ROUND
# =========================
print("\n" + "=" * 60)
print("STEP 6: Generate predictions on training corpus for next round")
print("=" * 60)

# Generate predictions on the ENTIRE training set for use in next round's filtering
print(f"Generating predictions on {len(reviews_df):,} training reviews...")

# Create dataset from full training corpus (for retrieval)
train_corpus_ds = Dataset.from_pandas(
    reviews_df[["text"]].reset_index(drop=True),
    preserve_index=False
).map(tok, batched=True, remove_columns=["text"])

# Get predictions (using the best model already loaded)
predictions_output = trainer.predict(train_corpus_ds)
predicted_labels = predictions_output.predictions.argmax(axis=-1)
predicted_probs = torch.softmax(torch.tensor(predictions_output.predictions), dim=-1).numpy()

# Create predictions DataFrame aligned with training corpus
predictions_df = pd.DataFrame({
    'review_id': reviews_df['review_id'].values,
    'predicted_label_id': predicted_labels,
    'predicted_label': [id2label[pred] for pred in predicted_labels],
    'true_label': reviews_df['label'].values,
    'confidence': predicted_probs.max(axis=-1),
})

# Add per-class probabilities
for label_id, label_name in id2label.items():
    predictions_df[f'prob_{label_name}'] = predicted_probs[:, label_id]

# Save predictions
predictions_path = f"predictions_{round_name}.csv"
predictions_df.to_csv(predictions_path, index=False)
print(f"✅ Saved predictions to {predictions_path}")

# Print prediction statistics
print(f"\nPrediction Statistics:")
print(f"  Predicted label distribution:")
print(predictions_df['predicted_label'].value_counts())
print(f"\n  Accuracy on training set: {(predictions_df['predicted_label_id'] == predictions_df['true_label'].map(label2id)).mean():.4f}")
print(f"  Mean confidence: {predictions_df['confidence'].mean():.4f}")

print(f"\n💡 NEXT ROUND: Use these cached predictions for filtering during TF-IDF")
print(f"   Set: use_model_filtering=True, prev_model_path='{output_dir}'")

print(f"\n💡 FINAL EVALUATION: Test on completely unseen data with:")
print(f"   python evaluate_holdout.py --model_path {output_dir} --test_csv reviews_test.csv")
print(f"\n⚠️  reviews_test.csv was NEVER seen during training or dev validation!")
