import os
import mmh3
import numpy as np
from typing import List
from collections import defaultdict
from itertools import combinations
import shutil

def get_ngrams(text: str, n: int) -> set:
    # Basic word tokenization (can be matched to whatever specific requirements, let's just use split)
    # Actually wait - we can just split by whitespace
    words = text.split()
    if len(words) < n:
        # If document is shorter than ngram size, the set is empty or we use the whole thing. Let's return just one tuple of all words if so.
        return set([tuple(words)]) if words else set()
    return set(tuple(words[i:i+n]) for i in range(len(words) - n + 1))

def minhash_deduplication(
    input_files: List[os.PathLike],
    output_directory: os.PathLike,
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
):
    os.makedirs(output_directory, exist_ok=True)
    
    docs = []
    # 1. Read files and generate ngrams
    for filepath in input_files:
        with open(filepath, 'rt', encoding='utf-8') as f:
            text = f.read()
            docs.append({
                'path': filepath,
                'ngrams': get_ngrams(text, ngrams),
                'signature': []
            })
            
    # 2. Compute minhash signatures
    for doc in docs:
        sig = []
        for i in range(num_hashes):
            min_val = float('inf')
            for ngram in doc['ngrams']:
                # Hash the ngram
                h = mmh3.hash(str(ngram).encode('utf-8'), seed=i, signed=False)
                if h < min_val:
                    min_val = h
            sig.append(min_val)
        doc['signature'] = sig
        
    # 3. LSH grouping
    rows_per_band = num_hashes // num_bands
    buckets = defaultdict(list)
    
    for doc_idx, doc in enumerate(docs):
        sig = doc['signature']
        for band_idx in range(num_bands):
            start = band_idx * rows_per_band
            end = start + rows_per_band
            band_tuple = tuple(sig[start:end])
            bucket_id = (band_idx, band_tuple)
            buckets[bucket_id].append(doc_idx)
            
    # 4. Find candidate pairs and verify Jaccard
    candidates = set()
    for doc_indices in buckets.values():
        if len(doc_indices) > 1:
            for pair in combinations(doc_indices, 2):
                # Always sort pair to avoid symmetric duplicates
                candidates.add(tuple(sorted(pair)))
                
    # 5. Connected components for duplicates
    # Adjacency list
    adj = {i: [] for i in range(len(docs))}
    for u, v in candidates:
        # compute exact jaccard
        set_u = docs[u]['ngrams']
        set_v = docs[v]['ngrams']
        if not set_u and not set_v:
            sim = 1.0
        elif not set_u or not set_v:
            sim = 0.0
        else:
            sim = len(set_u & set_v) / len(set_u | set_v)
            
        if sim >= jaccard_threshold:
            adj[u].append(v)
            adj[v].append(u)
            
    # Component extraction
    visited = set()
    to_keep = []
    
    for i in range(len(docs)):
        if i not in visited:
            # We found a new component, keep the first one we see (i)
            to_keep.append(i)
            # BFS/DFS to mark all connected as visited
            queue = [i]
            visited.add(i)
            while queue:
                curr = queue.pop(0)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
                        
    # 6. Copy kept files to output
    for idx in to_keep:
        src = docs[idx]['path']
        filename = os.path.basename(src)
        dst = os.path.join(output_directory, filename)
        shutil.copyfile(src, dst)
        
