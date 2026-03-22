import os
import hashlib
from collections import defaultdict
from typing import List, Set
from xopen import xopen

def exact_line_deduplication(input_files: List[os.PathLike], output_directory: os.PathLike):
    os.makedirs(output_directory, exist_ok=True)
    
    # First pass: Count frequencies of line hashes
    line_counts = defaultdict(int)
    for filepath in input_files:
        with xopen(filepath, 'rt') as f:
            for line in f:
                # We hash the line to save memory. 
                # (You can also strip it, but usually exact line dedup compares the exact string or stripped string)
                # We'll use exact string so whitespace matters, as per standard definition, but maybe we should use raw line bytes.
                h = hashlib.sha256(line.encode('utf-8')).digest()
                line_counts[h] += 1
                
    # Second pass: write out only lines with count == 1
    for filepath in input_files:
        filename = os.path.basename(filepath)
        out_filepath = os.path.join(output_directory, filename)
        with xopen(filepath, 'rt') as f_in, xopen(out_filepath, 'wt') as f_out:
            for line in f_in:
                h = hashlib.sha256(line.encode('utf-8')).digest()
                if line_counts[h] == 1:
                    f_out.write(line)
