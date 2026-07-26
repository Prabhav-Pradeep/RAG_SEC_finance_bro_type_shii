import sys
from pathlib import Path
import types

# Fix import path for running script directly
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# --- THE RAGAS BUG FIX ---
dummy = types.ModuleType("langchain_community.chat_models.vertexai")
dummy.ChatVertexAI = type("ChatVertexAI", (object,), {})
sys.modules["langchain_community.chat_models.vertexai"] = dummy
# -------------------------

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.run_config import RunConfig

# Import metrics
try:
    from ragas.metrics.collections import faithfulness, answer_relevancy
except ImportError:
    from ragas.metrics import faithfulness, answer_relevancy

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

from src.retrieval.retriever import ProductionRetriever
from src.generation.generator import GroundedGenerator

def run_evaluation():
    print("=== Initializing RAGAS Evaluation Pipeline ===")
    
    # 1. Initialize Pipeline Components
    retriever = ProductionRetriever()
    generator = GroundedGenerator(model_name="llama3.2")

    # 2. Define Golden Test Questions
    test_questions = [
        "What emerging risks are associated with artificial intelligence technologies for Apple?",
        "Does Apple discuss legal or regulatory operational risks in the document?",
        "What are Apple's main business products mentioned in the filing?"
    ]

    questions = []
    answers = []
    contexts = []

    print("\nRunning RAG pipeline over test dataset...")
    for q in test_questions:
        print(f"Evaluating query: '{q}'")
        
        # Retrieve context chunks
        hits = retriever.retrieve(q, final_top_k=3)
        extracted_contexts = [h["payload"]["page_content"] for h in hits]
        
        # Generate grounded answer
        gen_out = generator.generate_answer(q, hits)
        
        questions.append(q)
        answers.append(gen_out["answer"])
        contexts.append(extracted_contexts)

    # 3. Format Data into HuggingFace Dataset
    data_dict = {
        "user_input": questions,
        "response": answers,
        "retrieved_contexts": contexts
    }
    dataset = Dataset.from_dict(data_dict)

    # 4. Configure Local Evaluator Models with Ragas Wrappers
    print("\nLoading local evaluator models (Llama 3.2 + BGE Embeddings)...")
    raw_llm = ChatOllama(
        model="llama3.2", 
        temperature=0.0,
        request_timeout=300.0
    )
    raw_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    evaluator_llm = LangchainLLMWrapper(raw_llm)
    evaluator_embeddings = LangchainEmbeddingsWrapper(raw_embeddings)

    # 5. Instantiate Metric Objects (Ragas 0.2+ requires initialized instances)
    metrics_list = []
    for m in [faithfulness, answer_relevancy]:
        if isinstance(m, type):
            metrics_list.append(m())
        elif callable(m) and not hasattr(m, "name"):
            try:
                metrics_list.append(m())
            except Exception:
                metrics_list.append(m)
        else:
            metrics_list.append(m)

    # 6. Run RAGAS Evaluation
    print("\nComputing Faithfulness and Answer Relevancy metrics (sequential mode)...")
    results = evaluate(
        dataset=dataset,
        metrics=metrics_list,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=RunConfig(max_workers=1, timeout=300)
    )

    # 7. Display and Save Results
    print("\n" + "="*50)
    print("RAGAS EVALUATION METRICS REPORT")
    print("="*50)
    
    df = results.to_pandas()
    print(df)
    
    print("\n--- Aggregate Performance ---")
    for col in ["faithfulness", "answer_relevancy"]:
        if col in df.columns:
            print(f"Mean {col.replace('_', ' ').title()}: {df[col].mean():.2%}")
        
    output_path = "data/evaluation_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\nDetailed evaluation report saved to: {output_path}")

if __name__ == "__main__":
    run_evaluation()