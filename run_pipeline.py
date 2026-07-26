import os
import json
import logging

# Import all the modules we built
from src.ingestion.sec_downloader import get_CIK_ticker, get_company_info, get_tags, download_htms
from src.ingestion.html_parser import parse_sec_html
from src.ingestion.validation import validate_parsed_document
from src.ingestion.chunking_engine import hybrid_chunker

# Set up clean logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    target_tickers = ["AAPL"]  # You can add more tickers here later!
    
    # We will collect every final, validated chunk here
    all_final_chunks = []
    
    for ticker in target_tickers:
        logging.info(f"=== Starting Pipeline for {ticker} ===")
        
        # ---------------------------------------------------------
        # STEP 1: Download the Data
        # ---------------------------------------------------------
        cik = get_CIK_ticker(ticker)
        company_data = get_company_info(cik)
        
        # Grab only the 2 most recent filings for this test
        filing_links = get_tags(company_data)[:2] 
        
        # download_htms handles rate limiting and deduplication!
        saved_filepaths = download_htms(filing_links, cik)
        
        # ---------------------------------------------------------
        # STEP 2 & 3: Parse, Validate, and Chunk
        # ---------------------------------------------------------
        for filepath in saved_filepaths:
            logging.info(f"Processing file: {filepath}")
            
            # Create document-level metadata
            doc_metadata = {
                "ticker": ticker,
                "form": "10-K/Q",  # You can parse this from the filename later
                "year": "2023"     # You can parse this from the filename later
            }
            
            # Parse the HTML into blocks (keeps tables intact)
            parsed_blocks = parse_sec_html(filepath, doc_metadata)
            
            # Run the blocks through the Validation Gate
            is_valid, reason = validate_parsed_document(parsed_blocks, filepath)
            
            if not is_valid:
                logging.error(f"Skipping {filepath} - {reason}")
                continue # Skip to the next file if it's garbage
                
            # Run the hybrid chunker
            chunks = hybrid_chunker(parsed_blocks, chunk_size=400, chunk_overlap=50)
            all_final_chunks.extend(chunks)
            
            logging.info(f"Successfully generated {len(chunks)} chunks from {filepath}.")

    # ---------------------------------------------------------
    # STEP 4: Save the Final Data
    # ---------------------------------------------------------
    # We save this to disk so Phase 3 (Embeddings) can just load it 
    # instantly without waiting for the whole pipeline to run again.
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "final_chunks.json")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_final_chunks, f, indent=4)
        
    logging.info(f"=== PIPELINE COMPLETE ===")
    logging.info(f"Saved a total of {len(all_final_chunks)} chunks to {output_file}")

if __name__ == "__main__":
    main()