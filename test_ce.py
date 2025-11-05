"""
Sanity check for cross-encoder on M3 Mac.
Tests if cross-encoder can produce valid scores without NaN.
"""

import os
# Apply same environment settings as main code
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from sentence_transformers import CrossEncoder
import torch
import numpy as np

print("="*60)
print("Cross-Encoder Sanity Check")
print("="*60)

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

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")
print()

# Test 1: CPU
print("="*60)
print("TEST 1: CPU Device")
print("="*60)
device = "cpu"
print(f"Using device: {device}")
print()

model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
print(f"Loading model: {model_name}")

try:
    ce = CrossEncoder(model_name, device=device)
    print("✓ Model loaded successfully")
    print()
    
    # Test 1: Simple pairs
    print("Test 1: Simple query-document pairs")
    print("-" * 60)
    pairs = [
        ["plumber", "This plumber fixed my water leak quickly."],
        ["plumber water", "Water leak in kitchen fixed by plumber."],
        ["restaurant", "Great Mexican food and service."],
    ]
    
    for i, (query, doc) in enumerate(pairs, 1):
        print(f"  Pair {i}: '{query}' <-> '{doc[:50]}...'")
    
    try:
        scores = ce.predict(pairs, batch_size=8, show_progress_bar=False)
        print()
        print(f"Scores: {scores}")
        print(f"Type: {type(scores)}")
        print(f"Dtype: {scores.dtype if hasattr(scores, 'dtype') else 'N/A'}")
        
        # Check for NaN
        scores_array = np.array(scores)
        has_nan = np.any(np.isnan(scores_array))
        print(f"Contains NaN: {has_nan}")
        
        if has_nan:
            print("❌ TEST FAILED: Scores contain NaN")
        else:
            print("✓ TEST PASSED: No NaN detected")
            print(f"Score range: [{scores_array.min():.4f}, {scores_array.max():.4f}]")
    
    except Exception as e:
        print(f"❌ CrossEncoder.predict raised: {repr(e)}")
        import traceback
        traceback.print_exc()
    
    print()
    print("Test 2: Batch with multiple candidates")
    print("-" * 60)
    
    # Simulate what happens in get_top_tfidf_word
    query = "plumber"
    candidates = ["leak", "pipe", "water", "drain", "great"]
    docs = [
        "Fixed my kitchen sink leak very quickly.",
        "The plumber replaced all the old pipes in my house.",
        "Water heater installation was done professionally."
    ]
    
    all_pairs = []
    for word in candidates:
        expanded = f"{query} {word}"
        for doc in docs:
            all_pairs.append([expanded, doc[:1000]])  # Truncate like in main code
    
    print(f"Testing {len(candidates)} candidates x {len(docs)} docs = {len(all_pairs)} pairs")
    
    try:
        batch_scores = ce.predict(all_pairs, batch_size=32, show_progress_bar=False)
        print(f"Batch scores shape: {np.array(batch_scores).shape}")
        
        # Check for NaN
        scores_array = np.array(batch_scores)
        has_nan = np.any(np.isnan(scores_array))
        print(f"Contains NaN: {has_nan}")
        
        if has_nan:
            print(f"❌ TEST FAILED: {np.sum(np.isnan(scores_array))}/{len(scores_array)} scores are NaN")
            print(f"NaN indices: {np.where(np.isnan(scores_array))[0]}")
        else:
            print("✓ TEST PASSED: No NaN in batch")
            
            # Reshape to (candidates, docs)
            scores_per_candidate = scores_array.reshape(len(candidates), len(docs))
            avg_scores = scores_per_candidate.mean(axis=1)
            
            print()
            print("Average scores per candidate:")
            for i, word in enumerate(candidates):
                print(f"  {word}: {avg_scores[i]:.4f}")
    
    except Exception as e:
        print(f"❌ Batch prediction raised: {repr(e)}")
        import traceback
        traceback.print_exc()
    
    print()

except Exception as e:
    print(f"❌ Failed to load model on CPU: {repr(e)}")
    import traceback
    traceback.print_exc()

# Test 2: MPS if available
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    print()
    print("="*60)
    print("TEST 2: MPS Device (Apple Silicon)")
    print("="*60)
    device = "mps"
    print(f"Using device: {device}")
    print()
    
    try:
        ce_mps = CrossEncoder(model_name, device=device)
        print("✓ Model loaded successfully on MPS")
        print()
        
        # Test simple pairs on MPS
        print("Testing simple pairs on MPS...")
        pairs = [
            ["plumber", "This plumber fixed my water leak quickly."],
            ["restaurant", "Great Mexican food and service."],
        ]
        
        try:
            scores = ce_mps.predict(pairs, batch_size=8, show_progress_bar=False)
            print(f"Scores: {scores}")
            
            scores_array = np.array(scores)
            has_nan = np.any(np.isnan(scores_array))
            print(f"Contains NaN: {has_nan}")
            
            if has_nan:
                print("❌ MPS TEST FAILED: Scores contain NaN")
            else:
                print("✅ MPS TEST PASSED: No NaN detected")
                print(f"Score range: [{scores_array.min():.4f}, {scores_array.max():.4f}]")
        
        except Exception as e:
            print(f"❌ MPS prediction failed: {repr(e)}")
            import traceback
            traceback.print_exc()
    
    except Exception as e:
        print(f"❌ Failed to load model on MPS: {repr(e)}")
        import traceback
        traceback.print_exc()
else:
    print()
    print("MPS not available - skipping MPS test")

print()
print("="*60)
print("Sanity check complete!")
print("="*60)
print()
print("Recommendation:")
device_rec = choose_device()
print(f"  Use device: '{device_rec}'")
