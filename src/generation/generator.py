import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

class GroundedGenerator:
    # Change the default model to the one you just downloaded
    def __init__(self, model_name: str = "llama3.2"):
        """
        Initializes the generator using a local Ollama server.
        """
        # We still use the OpenAI library, but we trick it into talking to our local machine!
        self.client = OpenAI(
            base_url="http://localhost:11434/v1", 
            api_key="ollama" # Required field by the library, but Ollama ignores it
        )
        self.model_name = model_name

    def _build_system_prompt(self) -> str:
        """Constructs the strict guardrail system prompt."""
        return (
            "You are a strict, precise financial analyst assistant.\n"
            "Your task is to answer the user's question ONLY using the provided retrieved context blocks.\n\n"
            "STRICT RULES:\n"
            "1. Grounding: Answer strictly using facts found in the context blocks. Do NOT use outside knowledge.\n"
            "2. Citations: Every single factual assertion, number, or statement MUST be followed by an inline citation "
            "referencing the Chunk ID and Section. Format: [Chunk X | Item Y].\n"
            "3. Fallback/Refusal: If the provided context does not contain sufficient information to fully answer the question, "
            "explicitly state: 'I cannot answer this question based on the provided SEC context.' Do NOT attempt to guess or synthesize.\n"
            "4. Structure: Keep responses concise, clear, and organized using Markdown formatting."
        )

    def _format_context(self, retrieved_chunks: list[dict]) -> str:
        """Formats the retrieved chunks into a structured text block for the LLM context window."""
        formatted_blocks = []
        for i, item in enumerate(retrieved_chunks):
            payload = item.get("payload", {})
            content = payload.get("page_content", "")
            metadata = payload.get("metadata", {})
            
            doc_id = item.get("doc_id", i)
            ticker = metadata.get("ticker", "N/A")
            year = metadata.get("year", "N/A")
            section = metadata.get("section", "N/A")
            
            block_header = f"--- CONTEXT CHUNK {doc_id} | Ticker: {ticker} | Year: {year} | Section: {section} ---"
            formatted_blocks.append(f"{block_header}\n{content}\n")
            
        return "\n".join(formatted_blocks)

    def generate_answer(self, query: str, retrieved_chunks: list[dict]) -> dict:
        """
        Takes the user query and retrieved chunks, formats the prompt,
        and calls the LLM with strict guardrails.
        """
        if not retrieved_chunks:
            return {
                "answer": "I cannot answer this question based on the provided SEC context (no chunks retrieved).",
                "citations_used": []
            }

        formatted_context = self._format_context(retrieved_chunks)
        
        user_prompt = (
            f"RETRIEVED CONTEXT:\n{formatted_context}\n\n"
            f"USER QUESTION: {query}\n\n"
            "ANSWER (remember to cite sources like [Chunk X | Section Y]):"
        )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self._build_system_prompt()},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0 # Deterministic output to prevent hallucination
        )

        answer_text = response.choices[0].message.content

        return {
            "query": query,
            "answer": answer_text,
            "context_count": len(retrieved_chunks)
        }

# --- EXECUTION BLOCK (End-to-End Pipeline Test) ---
if __name__ == "__main__":
    # Import the retriever we built in Phase 4
    from src.retrieval.retriever import ProductionRetriever

    print("=== Testing End-to-End RAG Pipeline ===")
    
    # 1. Initialize Retriever and Generator
    retriever = ProductionRetriever()
    generator = GroundedGenerator()

    # 2. Define Test Query
    query = "What were Apple's primary business activities or financial risks outlined?"
    print(f"\nUser Query: {query}")

    # 3. Retrieve Context
    print("Retrieving context chunks...")
    retrieved_hits = retriever.retrieve(query, final_top_k=3)

    # 4. Generate Answer
    print("Generating grounded answer with GPT-4o-mini...")
    result = generator.generate_answer(query, retrieved_hits)

    print("\n" + "="*50)
    print("GENERATED RESPONSE:")
    print("="*50)
    print(result["answer"])
    print("="*50)