import re
import os
import pandas as pd
from io import StringIO
from unstructured.partition.html import partition_html

def parse_sec_html(filepath: str, doc_metadata: dict) -> list:
    """
    Parses HTML and attaches both document-level (Ticker/Year) 
    and chunk-level (Item 1A) metadata to every block.
    """
    print(f"Parsing: {filepath}...")
    elements = partition_html(filename=filepath)
    
    parsed_blocks = []
    current_item_section = "Unknown" # This acts as our running bookmark
    
    # Regex to catch headers like "ITEM 1A. RISK FACTORS" or "Item 1."
    item_pattern = re.compile(r"^item\s+([0-9a-z]+)\.", re.IGNORECASE)
    
    for element in elements:
        category = element.category
        clean_text = str(element).strip()
        
        # 1. Update our "bookmark" if we hit a new SEC Item section
        if category in ["Title", "NarrativeText"] and len(clean_text) < 100:
            match = item_pattern.match(clean_text)
            if match:
                current_item_section = f"Item {match.group(1).upper()}"
                
        # 2. Build the combined metadata dictionary for this specific block
        block_metadata = {
            "ticker": doc_metadata.get("ticker"),
            "year": doc_metadata.get("year"),
            "form": doc_metadata.get("form"),
            "section": current_item_section  # Dynamically assigned!
        }
        
        # 3. Handle Text
        if category in ["Title", "NarrativeText", "ListItem", "Text"] and len(clean_text) > 20:
            parsed_blocks.append({
                "type": "text",
                "content": clean_text,
                "metadata": block_metadata
            })
                
        # 4. Handle Tables
        elif category == "Table":
            table_html = element.metadata.text_as_html
            if table_html:
                try:
                    dfs = pd.read_html(StringIO(table_html))
                    if dfs:
                        df = dfs[0]
                        df.dropna(how="all", axis=0, inplace=True)
                        df.dropna(how="all", axis=1, inplace=True)
                        markdown_table = df.to_markdown(index=False)
                        
                        parsed_blocks.append({
                            "type": "table",
                            "content": markdown_table,
                            "metadata": block_metadata # The table gets tagged with the section too!
                        })
                except Exception:
                    continue

    return parsed_blocks
if __name__ == "__main__":
    test_file = "data/raw_filings/0000320193_000032019325000079_aapl-20250927.htm" 
    dummy_metadata = {
        "ticker": "AAPL",
        "year": "2023",
        "form": "10-K"
    }
    if os.path.exists(test_file):
        blocks = parse_sec_html(test_file,dummy_metadata)
        
        for i, block in enumerate(blocks[:5]):
                print(f"\n--- Block {i} ({block['type'].upper()}) ---")
                print(f"Metadata: {block['metadata']}")
                print(f"Content: {block['content'][:150]}...")
    else:
        print(f"File not found: {test_file}. Update the test_file variable!")