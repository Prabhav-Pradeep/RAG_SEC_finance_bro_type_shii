import sys
from pathlib import Path
import types

# 1. Fix paths and patch the Ragas bug so the server boots cleanly
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
dummy = types.ModuleType("langchain_community.chat_models.vertexai")
dummy.ChatVertexAI = type("ChatVertexAI", (object,), {})
sys.modules["langchain_community.chat_models.vertexai"] = dummy

from fastapi import FastAPI
from pydantic import BaseModel
from src.retrieval.retriever import ProductionRetriever
from src.generation.generator import GroundedGenerator

# 2. Initialize the Server App
app = FastAPI(title="SEC Financial RAG API", version="1.0")

# 3. Load the Engine into Memory (This happens once on startup)
print("Booting up SEC Engine...")
retriever = ProductionRetriever()
generator = GroundedGenerator(model_name="llama3.2")
print("Engine Online!")

# 4. Define the JSON Request Structure
class QueryRequest(BaseModel):
    query: str

# 5. Create the API Endpoint
@app.post("/ask")
async def ask_rag(request: QueryRequest):
    # Retrieve top 3 chunks
    hits = retriever.retrieve(request.query, final_top_k=3)
    
    # Generate grounded answer
    result = generator.generate_answer(request.query, hits)
    
    # Return formatted JSON response
    return {
        "question": request.query,
        "answer": result["answer"],
        "sources": [hit["payload"]["metadata"] for hit in hits]
    }