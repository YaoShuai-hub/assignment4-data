import os
import argparse
from cs336_data.dedup import exact_line_deduplication

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    
    # gather all txt files
    input_files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.endswith('.txt')]
    print(f"Found {len(input_files)} files to deduplicate.")
    
    exact_line_deduplication(input_files, args.output_dir)
    print(f"Exact line deduplication complete. Output stored in {args.output_dir}")
