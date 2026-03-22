import os
import concurrent.futures
from fastwarc.warc import ArchiveIterator, WarcRecordType
import tqdm

from cs336_data.extraction import extract_text_from_html_bytes
from cs336_data.langid import identify_language
from cs336_data.pii import mask_emails, mask_phone_numbers, mask_ips
from cs336_data.toxicity import classify_nsfw, classify_toxic_speech
from cs336_data.gopher import gopher_quality_filter
from cs336_data.quality import classify_quality
from cs336_data.dedup import exact_line_deduplication

def process_record(record_bytes: bytes) -> str | None:
    # 1. Extraction
    text = extract_text_from_html_bytes(record_bytes)
    if not text or not text.strip():
        return None
        
    # 2. Language ID (Keep English)
    lang, lang_score = identify_language(text)
    if lang != 'en':
        return None
        
    # 3. Gopher quality filter
    if not gopher_quality_filter(text):
        return None
        
    # 4. NSFW & Toxicity filter
    nsfw_pred, nsfw_score = classify_nsfw(text)
    if nsfw_pred == 'nsfw':
        return None
        
    toxic_pred, toxic_score = classify_toxic_speech(text)
    if toxic_pred == 'toxic':
        return None
        
    # 5. Quality Classifier
    qual_pred, qual_score = classify_quality(text)
    if qual_pred == 'cc': # Skip low quality cc predictions
        return None
        
    # 6. PII Masking
    text, _ = mask_emails(text)
    text, _ = mask_phone_numbers(text)
    text, _ = mask_ips(text)
    
    return text

def process_file(input_path: str, output_path: str):
    """Process a single WARC/WET file and output a text file with retained records."""
    try:
        with open(output_path, 'w', encoding='utf-8') as out_f:
            for record in ArchiveIterator(open(input_path, 'rb'), record_types=WarcRecordType.conversion):
                record_bytes = record.reader.read()
                processed_text = process_record(record_bytes)
                if processed_text:
                    # Write with a delimiter or just double newlines
                    out_f.write(processed_text + '\n\n')
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

def run_pipeline(input_dir: str, output_dir: str, num_workers: int = 8):
    os.makedirs(output_dir, exist_ok=True)
    input_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.warc.gz') or f.endswith('.wet.gz')]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for input_file in input_files:
            filename = os.path.basename(input_file)
            output_file = os.path.join(output_dir, filename.replace('.warc.gz', '.txt').replace('.wet.gz', '.txt'))
            futures.append(executor.submit(process_file, input_file, output_file))
            
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            future.result() # raises exception if occurred

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Common Crawl processing pipeline")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    
    run_pipeline(args.input_dir, args.output_dir, args.workers)
