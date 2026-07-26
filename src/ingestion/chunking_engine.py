from langchain_text_splitters import RecursiveCharacterTextSplitter

def hybrid_chunker(parsed_blocks: list, chunk_size: int = 400, chunk_overlap: int = 50) -> list:
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base", 
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    final_chunks = []
    
    for block in parsed_blocks:
        content = block.get("content", "")
        metadata = block.get("metadata", {})
        block_type = block.get("type", "")
        
        if not content:
            continue
            
        if block_type == "table":
            metadata_copy = metadata.copy()
            metadata_copy["chunk_type"] = "table"
            
            final_chunks.append({
                "page_content": content,
                "metadata": metadata_copy
            })
            
        elif block_type == "text":
            split_texts = text_splitter.split_text(content)
            
            for chunk in split_texts:
                metadata_copy = metadata.copy()
                metadata_copy["chunk_type"] = "text"
                
                final_chunks.append({
                    "page_content": chunk,
                    "metadata": metadata_copy
                })
                
    return final_chunks

# --- EXECUTION BLOCK (For Testing) ---
if __name__ == "__main__":
    mock_blocks = [
        {
            "type": "text", 
            "content": "Apple Inc. (AAPL) designs, manufactures and markets smartphones, personal computers, tablets, wearables and accessories. " * 20, 
            "metadata": {"ticker": "AAPL", "section": "Item 1"}
        },
        {
            "type": "table", 
            "content": "| Metric | 2023 | 2022 |\n| Revenue | $383B | $394B |", 
            "metadata": {"ticker": "AAPL", "section": "Item 8"}
        }
    ]
    
    chunks = hybrid_chunker(mock_blocks)
    
    print(f"Original blocks: {len(mock_blocks)}")
    print(f"Total chunks created: {len(chunks)}\n")
    
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i+1} ({chunk['metadata']['chunk_type']}) ---")
        print(f"Metadata: {chunk['metadata']}")
        print(f"Content: {chunk['page_content'][:100]}...\n")